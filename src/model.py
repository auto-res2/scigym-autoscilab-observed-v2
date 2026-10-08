"""候補 SBML の表現・構築・シミュレーション・定数フィット。LLM-AutoSciLab の仮説を SciGym の反応回復課題に写す部分。

仮説 = 不完全モデルに足す反応の集合。各反応は反応物・生成物・活性化因子・阻害因子と速度則の形を持ち、
速度定数などの自由定数は実験の時系列に対してフィットする。公式コードの仮説（Python の式）に当たるのがこの ReactionSet。

候補のシミュレーションは別プロセス（SimWorker）で行う。LLM が出した候補は硬い ODE になりやすく、roadrunner の
C ライブラリ内で止まると Python のシグナルでは戻せないので、親から時間切れを検知して kill し、その候補の結果は失敗扱いにする。
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from scigym.data.sbml import SBML  # scigym（tellurium 経由の roadrunner）を libsbml より先に読む: 逆順だと二重の libsbml で落ちる
from scigym.data.simulator import Simulator, run_simulation

import libsbml  # noqa: E402
import numpy as np  # noqa: E402
import roadrunner  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)  # 候補の積分失敗は想定内（残差で罰する）。CVODE のエラー行でログを埋めない

Kinetics = Literal["mass_action", "michaelis_menten", "hill"]
PARAM_LOG_BOUNDS = (-20.0, 20.0)  # 速度定数・飽和定数の自然対数の範囲
HILL_LOG_BOUNDS = (0.0, math.log(4.0))  # Hill 係数 n ∈ [1, 4]
MAX_FIT_POINTS = 50  # フィットと食い違い計算に使う時点数の上限（ODE 解法の時間は変わらない）
FAIL_RESIDUAL = 10.0  # シミュレーションが落ちた点の残差（log スケール）
SIM_TIMEOUT_BASE = 15.0  # 1 バッチの時間切れ（秒）= BASE + PER_JOB × 件数。通常は 1 回 0.3〜2 ms
SIM_TIMEOUT_PER_JOB = 0.02


class Reaction(BaseModel):
    reactants: list[str] = Field(default_factory=list, description="species ids consumed; empty = synthesis")
    products: list[str] = Field(default_factory=list, description="species ids produced; empty = degradation")
    activators: list[str] = Field(default_factory=list, description="species that speed the reaction up without being consumed")
    inhibitors: list[str] = Field(default_factory=list, description="species that slow the reaction down without being consumed")
    kinetics: Kinetics = Field("mass_action", description="mass_action | michaelis_menten (saturating in the first reactant) | hill (sigmoidal)")


class ReactionSet(BaseModel):
    reasoning: str = Field("", description="one or two sentences")
    reactions: list[Reaction] = Field(default_factory=list, max_length=12)


def skeleton(rs: ReactionSet) -> tuple:
    """骨格 = 反応集合（反応物・生成物・修飾子の組）。速度則の形と定数は含めない"""
    return tuple(sorted(
        (tuple(sorted(r.reactants)), tuple(sorted(r.products)), tuple(sorted(r.activators + r.inhibitors)))
        for r in rs.reactions
    ))


def skeleton_text(rs: ReactionSet) -> str:
    return "; ".join(
        " + ".join(r.reactants or ["∅"]) + " -> " + " + ".join(r.products or ["∅"])
        + ("".join(f" [+{a}]" for a in r.activators)) + ("".join(f" [-{i}]" for i in r.inhibitors)) + f" ({r.kinetics})"
        for r in rs.reactions
    )


def cluster_entropy(skeletons: list[tuple]) -> float:
    """骨格クラスタの分布の Shannon エントロピー（bit）"""
    counts = np.array(list(Counter(skeletons).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ------------------------------------------------------------------ simulation worker
def _simulate_one(sim: Simulator, params: dict, initial: dict, species: list[str], n_points: int, index) -> np.ndarray | None:  # noqa: ANN001
    try:
        sim.prepare_simulation()
        for pid, v in params.items():
            sim._rr.setValue(pid, float(v))
        for sid, v in initial.items():
            sim._rr.setInitConcentration(sid, float(v), forceRegenerate=False)
        res = run_simulation(sim._rr, observed_species=species, observed_parameters=[],
                             rm_concentration_brackets=True, sed_simulation=sim.simulation)
        y = np.array([res.result[s] for s in species], dtype=float).T
        if y.shape[0] != n_points or not np.all(np.isfinite(y)):
            return None
        return y if index is None else y[index]
    except Exception:
        return None


def _worker_main(conn, sedml: str, species: list[str], n_points: int) -> None:  # noqa: ANN001
    """子プロセス本体。候補 SBML の roadrunner を持ち、親から来たバッチを順に回す。親が死んだら自分も終わる"""
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)
    sims: dict[str, Simulator] = {}
    while True:
        if not conn.poll(1.0):
            if os.getppid() == 1:
                return
            continue
        msg = conn.recv()
        kind = msg[0]
        if kind == "quit":
            return
        if kind == "load":
            _, cid, sbml = msg
            try:
                sims[cid] = Simulator(SBML(sbml, sedml))
                conn.send(True)
            except Exception as exc:
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
        elif kind == "run":
            _, cid, jobs, index = msg
            sim = sims.get(cid)
            conn.send([None] * len(jobs) if sim is None else [_simulate_one(sim, p, i, species, n_points, index) for p, i in jobs])


class SimWorker:
    """候補のシミュレーションを別プロセスで行う。時間切れなら kill して失敗扱い、次回に作り直す"""

    def __init__(self, sedml: str, species: list[str], n_points: int):
        self.sedml, self.species, self.n_points = sedml, species, n_points
        self.sbml: dict[str, str] = {}
        self.loaded: set[str] = set()
        self.restarts = 0
        self.proc = None
        self._start()

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child, self.sedml, self.species, self.n_points), daemon=True)
        self.proc.start()
        child.close()
        self.loaded = set()

    def _restart(self) -> None:
        self.restarts += 1
        try:
            self.proc.kill()
            self.proc.join(5)
        except Exception:
            pass
        self._start()

    def _call(self, msg: tuple, timeout: float):  # noqa: ANN201
        self.conn.send(msg)
        if not self.conn.poll(timeout):
            raise TimeoutError
        return self.conn.recv()

    def load(self, cid: str, sbml: str) -> None:
        """候補を登録して読み込む。SBML が壊れていれば ValueError"""
        self.sbml[cid] = sbml
        try:
            r = self._call(("load", cid, sbml), 60.0)
        except (TimeoutError, EOFError, OSError) as exc:
            self._restart()
            raise ValueError(f"load failed: {type(exc).__name__}")
        if r is not True:
            raise ValueError(r[1])
        self.loaded.add(cid)

    def run(self, cid: str, jobs: list[tuple[dict, dict]], index=None) -> list[np.ndarray | None]:  # noqa: ANN001
        if not jobs:
            return []
        try:
            if cid not in self.loaded:
                r = self._call(("load", cid, self.sbml[cid]), 60.0)
                if r is not True:
                    return [None] * len(jobs)
                self.loaded.add(cid)
            return self._call(("run", cid, jobs, index), SIM_TIMEOUT_BASE + SIM_TIMEOUT_PER_JOB * len(jobs))
        except (TimeoutError, EOFError, OSError):  # 時間切れ、または子が死んだ（接続が切れた）
            self._restart()
            return [None] * len(jobs)

    def close(self) -> None:
        try:
            self.conn.send(("quit",))
            self.proc.join(5)
        except Exception:
            pass
        if self.proc is not None and self.proc.is_alive():
            self.proc.kill()


@dataclass
class System:
    """1 件分の問題: 不完全モデル、種の情報、SED-ML の時間格子、シミュレーション用の子プロセス"""

    incomplete: SBML
    sedml: str
    species: list[str]
    perturbable: list[str]  # 初期濃度を変えてよい種（boundary でも constant でもない）
    initial: dict[str, float]
    compartment_of: dict[str, str]
    default_compartment: str
    n_points: int
    worker: SimWorker

    @classmethod
    def from_incomplete(cls, incomplete: SBML, sedml: str) -> "System":
        m = incomplete.model
        species = incomplete.get_species_ids()
        perturbable = [s for s in incomplete.get_species_ids(floating_only=True) if not m.getSpecies(s).getConstant()]
        initial = incomplete.get_initial_concentrations()
        comp = {s: m.getSpecies(s).getCompartment() for s in species}
        default_comp = m.getCompartment(0).getId() if m.getNumCompartments() else ""
        probe = Simulator(incomplete)
        probe.prepare_simulation()
        n_points = probe.simulation.getNumberOfSteps() + 1
        return cls(incomplete, sedml, species, perturbable, initial, comp, default_comp, n_points, SimWorker(sedml, species, n_points))

    def fit_index(self) -> np.ndarray:
        return np.unique(np.linspace(0, self.n_points - 1, min(self.n_points, MAX_FIT_POINTS)).round().astype(int))

    def concentration_scale(self) -> float:
        pos = [v for v in self.initial.values() if v > 0]
        return float(np.median(pos)) if pos else 1.0

    def allowed_bounds(self, factor: tuple[float, float]) -> dict[str, tuple[float, float]]:
        """実験で設定してよい初期濃度の範囲。既定値 c の factor 倍。c = 0 の種は [0, 既定値の中央値]"""
        scale = self.concentration_scale()
        return {
            s: (factor[0] * self.initial[s], factor[1] * self.initial[s]) if self.initial.get(s, 0.0) > 0 else (0.0, scale)
            for s in self.perturbable
        }


_candidate_counter = Counter()


@dataclass
class Candidate:
    """ReactionSet を SBML にしたもの。自由定数は params に持ち、シミュレーションのたびに子プロセスの roadrunner に入れる"""

    reaction_set: ReactionSet
    system: System
    params: dict[str, float] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)  # param id -> "rate" | "saturation" | "hill"
    sbml_string: str = ""
    cid: str = ""
    fit_error: float = math.inf

    def __post_init__(self) -> None:
        self.sbml_string, self.params, self.kinds = build_sbml(self.reaction_set, self.system)
        _candidate_counter["n"] += 1
        self.cid = f"c{_candidate_counter['n']}"
        self.system.worker.load(self.cid, self.sbml_string)  # 壊れた SBML はここで ValueError

    @property
    def skeleton(self) -> tuple:
        return skeleton(self.reaction_set)

    def simulate_many(self, jobs: list[tuple[dict, dict]], index=None) -> list[np.ndarray | None]:  # noqa: ANN001
        """(定数, 初期濃度) の組ごとに時系列（index 行だけ）。落ちた・止まった組は None"""
        return self.system.worker.run(self.cid, jobs, index)

    def simulate(self, initial: dict[str, float] | None = None, params: dict[str, float] | None = None) -> np.ndarray | None:
        """全種の時系列 (n_points × n_species)。落ちたら None"""
        return self.simulate_many([(params or self.params, initial or {})])[0]

    def sbml_with_params(self) -> str:
        """フィット後の定数を書き込んだ SBML 文字列（提出用）"""
        doc = libsbml.readSBMLFromString(self.sbml_string)
        for pid, v in self.params.items():
            doc.getModel().getParameter(pid).setValue(float(v))
        return libsbml.writeSBMLToString(doc)


def build_sbml(rs: ReactionSet, system: System) -> tuple[str, dict[str, float], dict[str, str]]:
    """不完全モデルに反応を足した SBML と、その自由定数の初期値・種類"""
    doc = libsbml.readSBMLFromString(system.incomplete.to_string())
    model = doc.getModel()
    level3 = model.getLevel() >= 3
    scale = system.concentration_scale()
    params: dict[str, float] = {}
    kinds: dict[str, str] = {}

    def new_param(pid: str, value: float, kind: str) -> str:
        p = model.createParameter()
        p.setId(pid)
        p.setValue(value)
        p.setConstant(True)
        params[pid] = value
        kinds[pid] = kind
        return pid

    for i, r in enumerate(rs.reactions):
        rid = f"h_r{i}"
        rx = model.createReaction()
        rx.setId(rid)
        rx.setReversible(False)
        if level3:
            rx.setFast(False)
        for sid, n in Counter(r.reactants).items():
            ref = rx.createReactant()
            ref.setSpecies(sid)
            ref.setStoichiometry(float(n))
            if level3:
                ref.setConstant(True)
        for sid, n in Counter(r.products).items():
            ref = rx.createProduct()
            ref.setSpecies(sid)
            ref.setStoichiometry(float(n))
            if level3:
                ref.setConstant(True)
        for sid in dict.fromkeys(r.activators + r.inhibitors):
            mod = rx.createModifier()
            mod.setSpecies(sid)
        anchor = (r.reactants or r.products or r.activators or r.inhibitors or [None])[0]
        comp = system.compartment_of.get(anchor, system.default_compartment) if anchor else system.default_compartment
        terms = [comp] if comp else []
        terms.append(new_param(f"{rid}_k", 1.0, "rate"))
        reactant_counts = Counter(r.reactants)
        first = r.reactants[0] if r.reactants else None
        n_id = new_param(f"{rid}_n", 2.0, "hill") if r.kinetics == "hill" else None
        for sid, n in reactant_counts.items():
            if sid == first and r.kinetics in ("michaelis_menten", "hill"):
                K = new_param(f"{rid}_Km", scale, "saturation")
                terms.append(_saturating(sid, K, n_id))
                if n > 1:
                    terms.append(f"pow({sid}, {n - 1})")
            else:
                terms.append(sid if n == 1 else f"pow({sid}, {n})")
        for j, sid in enumerate(r.activators):
            if r.kinetics == "mass_action":
                terms.append(sid)
            else:
                terms.append(_saturating(sid, new_param(f"{rid}_Ka{j}", scale, "saturation"), n_id))
        for j, sid in enumerate(r.inhibitors):
            K = new_param(f"{rid}_Ki{j}", scale, "saturation")
            terms.append(f"({K} / ({K} + {sid}))" if n_id is None else f"(pow({K}, {n_id}) / (pow({K}, {n_id}) + pow({sid}, {n_id})))")
        kl = rx.createKineticLaw()
        kl.setMath(libsbml.parseL3Formula(" * ".join(terms)))
    return libsbml.writeSBMLToString(doc), params, kinds


def _saturating(sid: str, K: str, n_id: str | None) -> str:
    if n_id is None:
        return f"({sid} / ({K} + {sid}))"
    return f"(pow({sid}, {n_id}) / (pow({K}, {n_id}) + pow({sid}, {n_id})))"


@dataclass
class Observations:
    """正解モデルで行った実験: 初期濃度の設定と、返ってきた全種の時系列"""

    system: System
    initials: list[dict[str, float]] = field(default_factory=list)
    trajectories: list[np.ndarray] = field(default_factory=list)  # 各 (n_points × n_species)

    def add(self, initial: dict[str, float], y: np.ndarray) -> None:
        self.initials.append(initial)
        self.trajectories.append(np.asarray(y, dtype=float))

    def __len__(self) -> int:
        return len(self.initials)

    def eps(self) -> np.ndarray:
        """種ごとの log 変換の下駄: 観測の最大値の 1e-3"""
        if self.trajectories:
            top = np.max(np.stack(self.trajectories), axis=(0, 1))
        else:
            top = np.array([self.system.initial.get(s, 0.0) for s in self.system.species])
        return 1e-3 * np.abs(top) + 1e-12

    def log_targets(self, index: np.ndarray, which: list[int] | None = None) -> np.ndarray:
        eps = self.eps()
        rows = which if which is not None else range(len(self))
        return np.stack([np.log(self.trajectories[i][index] + eps) for i in rows])


def log_prediction(cand: Candidate, obs: Observations, index: np.ndarray, which: list[int], params: dict[str, float]) -> np.ndarray:
    """候補の予測の log 時系列 (len(which) × len(index) × n_species)。落ちた実験は NaN。実験分を 1 バッチで子プロセスに送る"""
    eps = obs.eps()
    out = np.full((len(which), len(index), len(cand.system.species)), np.nan)
    ys = cand.simulate_many([(params, obs.initials[i]) for i in which], index)
    for k, y in enumerate(ys):
        if y is not None:
            out[k] = np.log(np.maximum(y, 0.0) + eps)
    return out


def fit(cand: Candidate, obs: Observations, which: list[int] | None = None, max_nfev: int = 150, restarts: int = 1,
        rng: np.random.Generator | None = None) -> float:
    """自由定数を log 空間で最小二乗フィットし、cand.params を更新して log RMSE を返す"""
    if len(obs) == 0 or not cand.params:
        cand.fit_error = _error(cand, obs, which) if len(obs) else math.inf
        return cand.fit_error
    which = list(which if which is not None else range(len(obs)))
    index = cand.system.fit_index()
    target = obs.log_targets(index, which)
    ids = list(cand.params)
    lo = np.array([HILL_LOG_BOUNDS[0] if cand.kinds[p] == "hill" else PARAM_LOG_BOUNDS[0] for p in ids])
    hi = np.array([HILL_LOG_BOUNDS[1] if cand.kinds[p] == "hill" else PARAM_LOG_BOUNDS[1] for p in ids])

    def residual(x: np.ndarray) -> np.ndarray:
        pred = log_prediction(cand, obs, index, which, dict(zip(ids, np.exp(x))))
        r = pred - target
        r[~np.isfinite(r)] = FAIL_RESIDUAL
        return r.ravel()

    starts = [np.log(np.array([cand.params[p] for p in ids]))]
    rng = rng or np.random.default_rng(0)
    for _ in range(restarts):
        starts.append(np.array([rng.uniform(0.0, math.log(3.0)) if cand.kinds[p] == "hill" else rng.uniform(-6, 6) for p in ids]))
    best_x, best_cost = starts[0], math.inf
    for x0 in starts:
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(residual, x0, bounds=(lo, hi), max_nfev=max_nfev, method="trf")
        except Exception:
            continue
        if sol.cost < best_cost:
            best_x, best_cost = sol.x, sol.cost
    cand.params = dict(zip(ids, np.exp(best_x)))
    cand.fit_error = _error(cand, obs, which)
    return cand.fit_error


def _error(cand: Candidate, obs: Observations, which: list[int] | None = None) -> float:
    which = list(which if which is not None else range(len(obs)))
    index = cand.system.fit_index()
    r = log_prediction(cand, obs, index, which, cand.params) - obs.log_targets(index, which)
    r[~np.isfinite(r)] = FAIL_RESIDUAL
    return float(np.sqrt(np.mean(r ** 2)))


def log_r2(cand: Candidate, obs: Observations) -> float:
    """log 濃度の決定係数。種ごとの平均を基準にする（種の間でスケールが桁違いなので、全体平均だと見かけ上 1 に近づく）"""
    index = cand.system.fit_index()
    rows = list(range(len(obs)))
    P = log_prediction(cand, obs, index, rows, cand.params)
    T = obs.log_targets(index, rows)
    P = np.where(np.isfinite(P), P, T + FAIL_RESIDUAL)
    ss_res = float(np.sum((P - T) ** 2))
    ss_tot = float(np.sum((T - T.mean(axis=(0, 1), keepdims=True)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def bootstrap_confidence(cand: Candidate, obs: Observations, n_bootstrap: int, rng: np.random.Generator,
                         max_nfev: int = 60, min_r2: float = 0.5) -> tuple[float, list[dict[str, float]]]:
    """実験を再標本化して再フィットし、予測の正規化 std から確信度 = 1 − mean std（公式 learner.py の定義）。再フィットした定数の組も返す。
    公式と同じく R² < 0.5 の候補は明らかに外れているので確信度 0（安定していても Refine に進まない）"""
    if len(obs) < 2:
        return 0.0, []
    if log_r2(cand, obs) < min_r2:
        return 0.0, []
    index = cand.system.fit_index()
    allrows = list(range(len(obs)))
    refits: list[dict[str, float]] = []
    preds = []
    saved = dict(cand.params)
    for _ in range(n_bootstrap):
        rows = sorted(set(rng.integers(0, len(obs), len(obs)).tolist()))
        cand.params = dict(saved)
        fit(cand, obs, which=rows, max_nfev=max_nfev, restarts=0)
        refits.append(dict(cand.params))
        preds.append(log_prediction(cand, obs, index, allrows, cand.params))
    cand.params = saved
    cand.fit_error = _error(cand, obs)
    P = np.stack(preds)  # (B, n_exp, n_index, n_species)
    std = np.nanstd(P, axis=0)
    if not np.isfinite(std).any():
        return 0.0, refits
    return float(np.clip(1.0 - np.nanmean(std), 0.0, 1.0)), refits


class CandidateDistribution:
    """FalsificationSelector に渡す仮説分布: 候補点ごとに、候補 SBML 同士の予測（log 濃度）の分散を返す"""

    def __init__(self, members: list[tuple[Candidate, dict[str, float]]], obs: Observations, enabled: bool = True):
        self.members = members  # (候補, その定数)
        self.obs = obs
        self.enabled = enabled
        self.last_scores: np.ndarray | None = None

    def compute_disagreement(self, X: np.ndarray, param_names: list[str]) -> np.ndarray:
        n = X.shape[0]
        if not self.enabled or len(self.members) < 2:
            self.last_scores = np.zeros(n)
            return self.last_scores  # アブレーション: スコア 0 → select は多様性だけで選ぶ
        system = self.members[0][0].system
        index = system.fit_index()
        eps = self.obs.eps()
        logs = np.full((len(self.members), n, len(index), len(system.species)), np.nan)
        initials = [dict(zip(param_names, X[i].tolist())) for i in range(n)]
        for k, (cand, params) in enumerate(self.members):
            for i, y in enumerate(cand.simulate_many([(params, init) for init in initials], index)):
                if y is not None:
                    logs[k, i] = np.log(np.maximum(y, 0.0) + eps)
        var = np.nanvar(logs, axis=0)
        with np.errstate(all="ignore"):
            score = np.nanmean(var, axis=(1, 2))
        score[~np.isfinite(score)] = 0.0  # 落ちる候補点は選ばない
        self.last_scores = score
        return score
