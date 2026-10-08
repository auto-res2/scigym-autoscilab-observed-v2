"""1 件を LLM-AutoSciLab の閉ループ（Algorithm 1）で解く。GenHyp → Acquire → 照会 → RefineHyp → ConfGate を予算まで繰り返し、最良の候補 SBML を提出する。

公式コードから無改変で呼ぶのは EnsembleClient（small の K 並列サンプル）、LLMClient（large の構造化出力）、
FalsificationSelector.select（食い違い最大の点の貪欲多様選択）。SBML 固有の部分は src/model.py とこのファイル。
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from scigym.actions.experiment import Experiment
from scigym.api import ExperimentConfig
from scigym.data.question import Question

from autoscilab.al.falsification import FalsificationSelector
from autoscilab.llm.client import LLMClient
from autoscilab.llm.ensemble_client import EnsembleClient
from autoscilab.oracle.base import BaseOracle

from src.model import (Candidate, CandidateDistribution, Observations, ReactionSet, System, bootstrap_confidence,
                       cluster_entropy, fit, skeleton, skeleton_text)

MAX_HYPOTHESES = 8  # large が返す本命 + 対抗案の上限
LLM_RETRIES = 4


class Proposal(BaseModel):
    reasoning: str = Field("", description="two or three sentences")
    primary: ReactionSet = Field(description="the single most plausible complete reaction set")
    alternatives: list[ReactionSet] = Field(default_factory=list, description="2 to 6 competing complete reaction sets that the data cannot yet rule out")
    search_region: dict[str, list[float]] = Field(default_factory=dict, description="species id -> [low, high] initial concentration to explore in the next experiments; only species from the allowed list")


class SpeciesOracle(BaseOracle):
    """FalsificationSelector が見るオラクル役: 変えてよい種とその初期濃度の範囲だけ返す。正解モデルへの問い合わせは Experiment が行う"""

    def __init__(self, bounds: dict[str, tuple[float, float]]):
        self._bounds = bounds

    @property
    def domain(self) -> str:
        return "scigym"

    @property
    def parameter_names(self) -> list[str]:
        return list(self._bounds)

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return self._bounds

    @property
    def function_signature(self) -> str:
        return ""

    @property
    def param_description(self) -> str:
        return ""

    def run(self, params):  # noqa: ANN001
        raise NotImplementedError("experiments go through scigym Experiment.call_simulator")

    def evaluate_law(self, law_str):  # noqa: ANN001
        raise NotImplementedError


class TokenCounter:
    """公式クライアントが内部で持つ OpenAI クライアントの create を包んで usage を足し上げる（上流コードは変えない）"""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def _add(self, response) -> None:  # noqa: ANN001
        usage = getattr(response, "usage", None)
        self.calls += 1
        if usage is not None:
            self.input_tokens += usage.prompt_tokens or 0
            self.output_tokens += usage.completion_tokens or 0

    def wrap(self, completions) -> None:  # noqa: ANN001
        original = completions.create
        if inspect.iscoroutinefunction(original):
            async def create(*a, **kw):  # noqa: ANN002, ANN003
                r = await original(*a, **kw)
                self._add(r)
                return r
        else:
            def create(*a, **kw):  # noqa: ANN002, ANN003
                r = original(*a, **kw)
                self._add(r)
                return r
        completions.create = create


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Loop:
    def __init__(self, cfg: dict, out: Path):
        self.cfg = cfg
        self.out = out
        self.rng = np.random.default_rng(0)
        inst = cfg["instance_dir"]
        q = Question(sbml_directory_path=inst, task_difficulty="fully_observable")
        self.complete, self.incomplete = q.get_original_sbml(), q.get_partial_sbml()
        self.system = System.from_incomplete(self.incomplete, Path(inst, "truth.sedml").read_text())
        self.allowed = self.system.allowed_bounds(tuple(cfg["concentration_bounds_factor"]))
        self.obs = Observations(self.system)
        self.n_experiments = 0
        self.tokens = TokenCounter()
        key = os.environ["VERCEL_AI_GATEWAY_API_KEY"]
        self.small = EnsembleClient(base_url=cfg["base_url"], model=cfg["small_model"], k=cfg["ensemble_k"],
                                    temperature=cfg["ensemble_temperature"], max_tokens=cfg["ensemble_max_tokens"], api_key=key)
        self.large = LLMClient(model=cfg["large_model"], base_url=cfg["base_url"], api_key=key,
                               max_completion_tokens=cfg["large_max_tokens"])
        self.tokens.wrap(self.small._client.chat.completions)
        self.tokens.wrap(self.large._oa_client.chat.completions)
        self.candidates: dict[tuple, Candidate] = {}  # 骨格 -> これまでにフィットした候補（メモリ）
        self.rejected: list[str] = []
        self.trace: list[dict] = []
        self.best: Candidate | None = None

    # ------------------------------------------------------------ oracle
    def experiment(self, initial: dict[str, float]) -> bool:
        """正解モデルへの問い合わせ。予算を 1 消費する"""
        self.n_experiments += 1
        exp = Experiment(true_sbml=copy.deepcopy(self.complete), inco_sbml=copy.deepcopy(self.incomplete))
        actions = [f'change_initial_concentration("{s}", {v})' for s, v in initial.items()]
        res = exp.call_simulator(ExperimentConfig(experiment_action=actions, observed_species=[]))
        if not res.success or res.result is None:
            log(f"experiment {self.n_experiments} failed: {res.error_message[:120]}")
            return False
        y = np.array([res.result[s] for s in self.system.species], dtype=float).T
        self.obs.add(initial, y)
        return True

    # ------------------------------------------------------------ prompts
    def context(self) -> str:
        sp = "\n".join(f"- {s}: initial concentration {self.system.initial.get(s, 0.0):.4g}, compartment {self.system.compartment_of[s]}" for s in self.system.species)
        lines = [
            "# Task",
            "The SBML model below lists species and compartments, but all of its reactions were removed.",
            "Propose the complete set of reactions (with kinetics) that reproduces the measured dynamics of the true system.",
            "Use only the species ids listed. A reaction may have no reactants (synthesis) or no products (degradation).",
            "Kinetics: mass_action = rate proportional to the product of reactant concentrations (activators multiply the rate, inhibitors divide it);",
            "michaelis_menten = saturating in the first reactant (and in activators); hill = sigmoidal in the first reactant (and in activators).",
            "Rate constants and saturation constants are fitted automatically afterwards, so do not give numbers.",
            "# Species", sp,
            f"# Experiments so far ({len(self.obs)} of {self.cfg['budget']} budget used: {self.n_experiments})",
        ]
        idx = np.unique(np.linspace(0, self.system.n_points - 1, 5).round().astype(int))
        for i, (init, y) in enumerate(zip(self.obs.initials, self.obs.trajectories)):
            pert = ", ".join(f"{s}={v:.4g}" for s, v in init.items()) or "default initial concentrations"
            lines.append(f"## Experiment {i + 1}: {pert}")
            lines.append("species: " + " | ".join(f"t[{j}]" for j in idx))
            for k, s in enumerate(self.system.species):
                lines.append(f"{s}: " + " | ".join(f"{y[j, k]:.3g}" for j in idx))
        if self.candidates:
            ranked = sorted(self.candidates.values(), key=lambda c: c.fit_error)
            lines.append("# Candidate reaction sets evaluated so far (fit error = RMSE of log concentrations over all experiments; lower is better)")
            lines += [f"- error {c.fit_error:.3f}: {skeleton_text(c.reaction_set)}" for c in ranked[:8]]
        if self.rejected:
            lines.append("# Reaction sets that fit poorly (avoid repeating them)")
            lines += [f"- {t}" for t in self.rejected[-6:]]
        return "\n".join(lines)

    # ------------------------------------------------------------ GenHyp
    def sample_small(self, prompt: str) -> list[ReactionSet]:
        """Algorithm 2: K ずつ並列サンプルし、骨格クラスタのエントロピーが落ち着くか K_max で止める"""
        system_prompt = "You are a systems biologist proposing mechanistic hypotheses for a dynamical model of a biological system."
        messages = [{"role": "user", "content": prompt + "\n\nPropose ONE complete reaction set."}]
        pool: list[ReactionSet] = []
        h_prev = None
        k, k_max, tau = self.cfg["ensemble_k"], self.cfg["ensemble_k_max"], self.cfg["ensemble_stability_threshold"]
        for attempt in range(k_max // k + 3):
            if len(pool) >= k_max:
                break
            try:  # b = min(K, K_max − |C|)
                batch = self.small.complete_json_k(messages, system=system_prompt, schema=ReactionSet, min_valid=1,
                                                   k_override=min(k, k_max - len(pool)))
            except RuntimeError as exc:  # 1 つも JSON にならなかった: 少し待って取り直す
                log(f"small batch {attempt}: {exc}")
                time.sleep(5)
                continue
            pool += [rs for rs in batch if self.valid(rs)]
            log(f"small batch {attempt}: {len(batch)} parsed, pool {len(pool)}")
            if len(pool) < k:
                continue
            h = cluster_entropy([skeleton(rs) for rs in pool])
            if h_prev is not None and abs(h - h_prev) < tau:
                break
            h_prev = h
        return pool

    def valid(self, rs: ReactionSet) -> bool:
        """種 id が存在し、反応が空でなく、既定条件でシミュレーションが通る"""
        if not rs.reactions:
            return False
        known = set(self.system.species)
        for r in rs.reactions:
            if not (r.reactants or r.products) or not set(r.reactants + r.products + r.activators + r.inhibitors) <= known:
                return False
        return self.candidate(rs) is not None

    def candidate(self, rs: ReactionSet) -> Candidate | None:
        key = skeleton(rs)
        if key in self.candidates:
            return self.candidates[key]
        try:
            cand = Candidate(rs, self.system)
        except Exception:
            return None
        if cand.simulate() is None:
            return None
        self.candidates[key] = cand
        return cand

    def synthesize(self, prompt: str, pool: list[ReactionSet]) -> Proposal | None:
        clusters: dict[tuple, list[ReactionSet]] = {}
        for rs in pool:
            clusters.setdefault(skeleton(rs), []).append(rs)
        ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        lines = ["# Hypotheses sampled by a smaller model (grouped by reaction structure, with how many samples proposed each)"]
        for key, members in ranked[:MAX_HYPOTHESES * 2]:
            err = self.candidates[key].fit_error if key in self.candidates else math.inf
            lines.append(f"- {len(members)} samples{'' if math.isinf(err) else f', fit error {err:.3f}'}: {skeleton_text(members[0])}")
        allowed = ", ".join(f"{s} in [{lo:.4g}, {hi:.4g}]" for s, (lo, hi) in self.allowed.items())
        task = (
            "\n\nSynthesize these into a primary hypothesis and 2 to 6 alternatives that the experiments so far cannot distinguish "
            "(you may merge, correct or extend the sampled sets). Then choose a search region: the species whose initial concentrations the next "
            f"experiments should vary, each with a [low, high] range inside the allowed ranges: {allowed}."
        )
        system_prompt = "You are a senior systems biologist directing an automated discovery loop over a dynamical model."
        messages = [{"role": "user", "content": prompt + "\n\n" + "\n".join(lines) + task}]
        for attempt in range(LLM_RETRIES):
            try:
                return self.large.complete_json(messages, system=system_prompt, schema=Proposal)
            except Exception as exc:
                log(f"large synthesis attempt {attempt}: {type(exc).__name__}: {str(exc)[:160]}")
                time.sleep(10)
        return None

    def gen_hyp(self) -> tuple[list[Candidate], dict[str, tuple[float, float]]]:
        prompt = self.context()
        pool = self.sample_small(prompt)
        proposal = self.synthesize(prompt, pool)
        hyps: list[Candidate] = []
        if proposal is not None:
            for rs in [proposal.primary] + proposal.alternatives:
                c = self.candidate(rs) if self.valid(rs) else None
                if c is not None and c not in hyps:
                    hyps.append(c)
        for rs in pool:  # large が落ちた・足りないときは small のクラスタで埋める
            c = self.candidates.get(skeleton(rs))
            if len(hyps) >= MAX_HYPOTHESES:
                break
            if c is not None and c not in hyps:
                hyps.append(c)
        if self.best is not None and self.best not in hyps:  # メモリ: これまでの最良は常に残す
            hyps.insert(0, self.best)
        region = self.region(proposal.search_region if proposal else {})
        return hyps[:MAX_HYPOTHESES], region

    def region(self, proposed: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
        region = {}
        for s, rng_ in proposed.items():
            if s in self.allowed and isinstance(rng_, list) and len(rng_) == 2:
                lo, hi = self.allowed[s]
                a, b = max(lo, min(hi, float(rng_[0]))), max(lo, min(hi, float(rng_[1])))
                if b > a:
                    region[s] = (a, b)
        return region or dict(self.allowed)

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        budget = int(self.cfg["budget"])
        per_iter = int(self.cfg["experiments_per_iter"])
        n_boot = int(self.cfg["n_bootstrap"])
        self.experiment({})  # 最初の 1 回は既定条件の観測
        confidence, refits = 0.0, []
        round_no = 0
        while self.n_experiments < budget and round_no < 2 * budget:  # 仮説が一度も立たない件で回り続けない
            round_no += 1
            hyps, region = self.gen_hyp()
            if not hyps:
                log("no valid hypothesis this round")
                continue
            for c in hyps:
                fit(c, self.obs, rng=self.rng)
            hyps.sort(key=lambda c: c.fit_error)
            leader = hyps[0]
            confidence, refits = bootstrap_confidence(leader, self.obs, n_boot, self.rng)
            refine = confidence >= float(self.cfg["confidence_threshold"])
            if refine and len(refits) >= 2:
                members = [(leader, p) for p in refits]  # Refine: 本命の再フィット同士が最も割れる点
            else:
                members = [(c, c.params) for c in hyps]
            dist = CandidateDistribution(members, self.obs, enabled=self.cfg["acquisition"] == "disagreement")
            selector = FalsificationSelector(SpeciesOracle(region), dist, n_candidates=int(self.cfg["n_candidates"]),
                                             diversity_weight=float(self.cfg["diversity_weight"]))
            points = selector.select(n=per_iter, rng=self.rng)  # 常に per_iter 点を選び、予算の残りだけ使う
            chosen = points[: budget - self.n_experiments]
            for p in chosen:
                self.experiment(p)
            for c in hyps:
                fit(c, self.obs, restarts=0, rng=self.rng)
            hyps.sort(key=lambda c: c.fit_error)
            self.best = min([self.best, hyps[0]] if self.best else [hyps[0]], key=lambda c: c.fit_error)
            self.rejected += [skeleton_text(c.reaction_set) for c in hyps if c.fit_error > 3 * hyps[0].fit_error and skeleton_text(c.reaction_set) not in self.rejected]
            self.trace.append({
                "round": round_no, "mode": "refine" if refine else "disambiguate", "confidence": confidence,
                "hypotheses": [{"skeleton": skeleton_text(c.reaction_set), "fit_error": c.fit_error} for c in hyps],
                "region": {s: list(b) for s, b in region.items()}, "experiments": chosen,
                "max_disagreement": float(np.nanmax(dist.last_scores)) if dist.last_scores is not None else None,
                "n_experiments": self.n_experiments, "tokens": [self.tokens.input_tokens, self.tokens.output_tokens],
            })
            log(f"round {round_no}: {len(hyps)} hyps, best {hyps[0].fit_error:.3f} ({skeleton_text(hyps[0].reaction_set)[:80]}), "
                f"confidence {confidence:.2f} {'refine' if refine else 'disambiguate'}, experiments {self.n_experiments}/{budget}")
        if self.best is not None:
            fit(self.best, self.obs, rng=self.rng)
        self.finish(confidence)
        self.system.worker.close()

    def finish(self, confidence: float) -> None:
        if self.best is not None:
            (self.out / "final_model.xml").write_text(self.best.sbml_with_params())
        (self.out / "trace.json").write_text(json.dumps(self.trace, indent=1, default=float))
        (self.out / "tokens.json").write_text(json.dumps({
            "input_tokens": self.tokens.input_tokens, "output_tokens": self.tokens.output_tokens, "llm_calls": self.tokens.calls,
            "n_experiments": self.n_experiments, "n_rounds": len(self.trace),
            "sim_worker_restarts": self.system.worker.restarts,  # ODE が止まって子プロセスを作り直した回数
            "best_fit_error": self.best.fit_error if self.best else None, "final_confidence": confidence,
            "submitted": self.best is not None,
        }))
        log(f"done: submitted={self.best is not None} experiments={self.n_experiments} tokens={self.tokens.input_tokens}/{self.tokens.output_tokens}")


def main() -> None:
    cfg = json.loads(sys.argv[1])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Loop(cfg, out).run()


if __name__ == "__main__":
    main()
