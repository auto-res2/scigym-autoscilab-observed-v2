"""1 つの run（= 1 手法 × 1 split）で SciGym-small の各件を解かせ、評価層 scigym_<split> の入力ファイルを書く。

method: react     → src.train（SciGym 公式 Controller）
method: autoscilab → src.inference（LLM-AutoSciLab の SciGym 用ループ）
件ごとに 1 プロセス。モデルは Vercel AI Gateway 経由。
"""

import gzip
import json
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.preprocess import instances

BUDGET_EXCEEDED = 42
CONTEXT_OVERFLOW = 43
EXECUTOR = {"react": "src.train", "autoscilab": "src.inference"}
SANITY_REACT_ITERATIONS = 2
SANITY_AUTOSCILAB_BUDGET = 9  # 観測 1 + 4 + 4: 2 反復回って確信度ゲートまで通る最小


def cli_args():
    """hydra 形式の key=value を読む。hydra / omegaconf は scigym が固定する petab の antlr 版と衝突する"""
    return {k: yaml.safe_load(v) for k, _, v in (a.partition("=") for a in sys.argv[1:])}


def instance_args(cfg, instance, out):
    args = {"instance_dir": str(instance), "out_dir": str(out), "base_url": cfg.base_url}
    if cfg.method == "react":
        args |= {k: cfg.run[k] for k in ("model", "max_iterations", "eval_debug_rounds", "temperature", "max_tokens")}
        if cfg.mode == "sanity":
            args["max_iterations"] = SANITY_REACT_ITERATIONS
    else:
        args |= {k: cfg.run[k] for k in (
            "large_model", "small_model", "budget", "experiments_per_iter", "acquisition", "ensemble_k", "ensemble_k_max",
            "ensemble_stability_threshold", "ensemble_temperature", "ensemble_max_tokens", "large_max_tokens", "n_candidates",
            "diversity_weight", "confidence_threshold", "n_bootstrap", "concentration_bounds_factor")}
        if cfg.mode == "sanity":
            args["budget"] = SANITY_AUTOSCILAB_BUDGET
    return args


def run_instance(cfg, run_dir, instance):
    out = run_dir / "instances" / instance.name
    if (out / "tokens.json").exists():
        return 0
    if (out / "context_overflow").exists():
        return 0  # 文脈長を超えた件。やり直しても同じなので「提出なし」として不完全モデルで採点する
    if (out / "stdout.txt").exists() and (out / "stdout.txt").read_text().count("killed after") >= 3:
        return 0  # 3 試行とも上限で打ち切られた件。公式の「有効な提出なし」と同じく不完全モデルで採点する
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "stdout.txt"
    with open(log_path, "a") as log:
        proc = subprocess.Popen([sys.executable, "-m", EXECUTOR[cfg.method], json.dumps(instance_args(cfg, instance, out))],
                                stdout=log, stderr=subprocess.STDOUT)
        started = time.time()
        while proc.poll() is None:
            time.sleep(30)
            # LLM のコードや提出モデルの評価が ODE の C ライブラリ内で止まると公式の 3 分制限（SIGALRM）が効かない。
            # 両方の executor は反復ごとに出力するので、出力が止まったままの試行は打ち切って最初からやり直す
            silent = time.time() - log_path.stat().st_mtime
            if time.time() - started > cfg.instance_timeout or silent > cfg.stall_timeout:
                proc.kill()
                proc.wait()
                print(f"[{instance.name}] killed after {int(time.time() - started)}s ({int(silent)}s without output)", file=log)
                return 1
    if proc.returncode == CONTEXT_OVERFLOW:
        (out / "context_overflow").touch()
    if not (out / "tokens.json").exists():  # 失敗した run の作業ディレクトリは残らないので原因を標準出力へ
        print(f"[{instance.name}] no tokens.json; log tail:", *(out / "stdout.txt").read_text().splitlines()[-25:], sep="\n  ")
    if (out / "tokens.json").exists() or (out / "context_overflow").exists():
        checkpoint(run_dir, instance.name)
    return proc.returncode


def checkpoint(run_dir, name):
    """終わった件を 1 件 1 書庫で残す。job が上限で打ち切られても、集めた書庫から続きを実行できる"""
    out = run_dir / "instances" / name
    shutil.rmtree(out / "codes", ignore_errors=True)  # 反復ごとのコードは chat_history.yaml に含まれる
    (out / "chat_history_readable.txt").unlink(missing_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with tarfile.open(run_dir / "checkpoints" / f"{name}.tar.gz", "w:gz") as tar:
        tar.add(out, arcname=f"instances/{name}")


def main():
    cli = cli_args()
    run_id = cli["run"]
    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg.update(cli)
    cfg["run"] = yaml.safe_load(open(f"config/run/{run_id}.yaml"))
    split = cfg["run"]["split"]
    cfg.update({k: v for k, v in cfg["run"].items() if k in ("workers", "instance_timeout", "stall_timeout")})  # run ごとの上書き
    cfg = SimpleNamespace(**cfg, method=cfg["run"]["method"], task=f"scigym_{split}")
    run_dir = Path(cfg.results_dir) / run_id
    todo = instances(cfg.data_root, split, cfg.mode)
    stage = cfg.mode.upper()
    # 打ち切った前 run の件ごとの書庫（リポジトリの resume/<run_id>/）から続きを実行する
    if not (run_dir / "instances").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        for archive in sorted(Path("resume", run_id).glob("**/*.tar.gz")):
            with tarfile.open(archive) as tar:
                tar.extractall(run_dir)
    # API エラーで tokens.json が出なかった件は 2 回までやり直す。予算超過（402）は即座に run を止める
    for _ in range(3):
        with ThreadPoolExecutor(cfg.workers) as pool:
            codes = list(pool.map(lambda p: run_instance(cfg, run_dir, p), todo))
        if BUDGET_EXCEEDED in codes:
            print(f"{stage}_VALIDATION: FAIL reason=budget_exceeded")
            sys.exit(1)
    submitted, totals = {}, {"input_tokens": 0, "output_tokens": 0}
    n_experiments, n_overflow = [], 0
    for instance in todo:
        out = run_dir / "instances" / instance.name
        if (out / "context_overflow").exists():
            n_overflow += 1
        if not (out / "tokens.json").exists():
            continue  # 文脈超過か、3 試行とも終わらなかった件: 提出無しとして評価層に渡す
        info = json.loads((out / "tokens.json").read_text())
        submitted[instance.name] = (out / "final_model.xml").read_text() if (out / "final_model.xml").exists() else None
        n_experiments.append(info["n_experiments"])
        for k in totals:
            totals[k] += info[k]
    totals |= {
        "mean_experiments": sum(n_experiments) / len(n_experiments) if n_experiments else 0.0,
        "n_with_submission": sum(v is not None for v in submitted.values()),
        "n_context_overflow": n_overflow,
        "n_not_finished": len(todo) - len(submitted),
    }
    (run_dir / "eval_inputs").mkdir(parents=True, exist_ok=True)
    # 評価入力は gzip で書く（参照 SBML と提出 SBML を全件持つので大きい）
    with gzip.open(run_dir / "eval_inputs" / f"{cfg.task}.json.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"instances": [{
            "id": p.name,
            "reference_sbml": (p / "truth.xml").read_text(),
            "incomplete_sbml": (p / "partial.xml").read_text(),
            "reference_sedml": (p / "truth.sedml").read_text(),
            "submitted_sbml": submitted.get(p.name),
        } for p in todo]}))
    (run_dir / "tokens.json").write_text(json.dumps(totals))
    # 取り込みは 1 ファイル 1 API 呼び出しなので、件ごとの出力は 1 つの書庫にまとめる（GitHub の secondary rate limit 対策）
    with tarfile.open(run_dir / "instances.tar.gz", "w:gz") as tar:
        tar.add(run_dir / "instances", arcname="instances")
    shutil.rmtree(run_dir / "instances")
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)  # 全件そろったので件ごとの書庫は不要
    print(f"{stage}_VALIDATION_SUMMARY: {json.dumps({'n_instances': len(todo), **totals})}")
    if totals["n_with_submission"] == 0:
        print(f"{stage}_VALIDATION: FAIL reason=no_submission")
        sys.exit(1)
    print(f"{stage}_VALIDATION: PASS")


if __name__ == "__main__":
    main()
