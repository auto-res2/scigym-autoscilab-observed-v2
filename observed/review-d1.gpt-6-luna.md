# verify_implementation の試行（airas PR feat/implementation-review）

- 入力: scigym-autoscilab の record.json（d1 の宣言、仮説・claim・design・params と引用 passage 本文）、結果コミット ef51f93253ba の src/ config/ Dockerfile、observed.json は sanity run 37806579307（version 3、values は上位 10 件）。
- モデル: vercel_ai_gateway/openai/gpt-6-luna。51 秒。
- 以下はモデルの出力そのまま（kind / where / statement / 証拠）。

## contradiction（失敗）

- `src/model.py Reaction.kinetics` — 宣言では候補速度則を mass-action または Michaelis–Menten としているが、コードは hill も候補空間に含める。
  証拠: `Literal["mass_action", "michaelis_menten", "hill"]`
- `src/model.py CandidateDistribution.compute_disagreement` — 宣言は全種・全時点の対数濃度分散を使うとしているが、コードは最大50時点に間引いた値で食い違いを計算する。
  証拠: `index = system.fit_index()`; `MAX_FIT_POINTS = 50`; observed: 10001時点に対し fit 用 `linspace` は50点
- `src/model.py System.allowed_bounds` — 宣言された初期濃度範囲は既定値の0.2〜5倍だが、既定値が0の種にはコードが0〜正の中央値を許す。
  証拠: `else (0.0, scale)`; observed: 既定値0の種の範囲が `[0.0, 100.0]`
- `src/model.py bootstrap_confidence` — 宣言は正規化stdに基づく確信度としているが、コードは予測のstdの単純平均を1から引いてクリップし、正規化していない。
  証拠: `1.0 - np.nanmean(std)`; 宣言: `確信度 = 1 − 正規化 std`

## unverified（失敗）

- `observed __main__.cli_args / src.main.instance_args` — 宣言された autoscilab-luna の full 実行（budget=20、SciGym-small 全件）は観測されておらず、観測は sanity 実行（budget=9、1件）に限られる。
  証拠: observed: `mode: sanity`; 子プロセス引数 `budget: 9`; `instances` の戻り値 `len: 1`

## undeclared（報告）

- `src/model.py ReactionSet.reactions` — 候補反応集合は最大12反応に制限されている。
  証拠: `reactions: list[Reaction] = Field(default_factory=list, max_length=12)`
- `src/inference.py Loop.context` — 各実験の全時系列ではなく、5つの等間隔時点の濃度だけを LLM に提示する。
  証拠: `np.linspace(0, self.system.n_points - 1, 5)`
- `src/model.py fit` — 自由定数のフィットは対数濃度のRMSEを目的関数とし、候補ごとに初期値のほか最大1回のランダム再始動を使う。
  証拠: `pred - target`（対数値）を残差にし、`restarts: int = 1`; `least_squares(..., max_nfev=max_nfev)`
- `src/model.py PARAM_LOG_BOUNDS / HILL_LOG_BOUNDS` — フィット対象の速度・飽和定数は自然対数で−20〜20に制限され、Hill係数は1〜4に制限される。
  証拠: `PARAM_LOG_BOUNDS = (-20.0, 20.0)`; `HILL_LOG_BOUNDS = (0.0, math.log(4.0))`
- `src/inference.py Loop.valid` — 候補は反応が空でないこと、参照種が既知であることに加えて、既定条件でシミュレーションが通る場合だけ採用される。
  証拠: `if not rs.reactions: return False`; `return self.candidate(rs) is not None`
- `src/inference.py Loop.__init__` — 実験選択やブートストラップ等に使う乱数生成器はseed 0で初期化される。
  証拠: `self.rng = np.random.default_rng(0)`
- `src/inference.py Loop.gen_hyp` — largeモデルの提案がない、または候補が不足する場合、smallモデルのサンプル候補で仮説集合を補い、過去の最良候補も集合に残す。
  証拠: `for rs in pool:  # large が落ちた・足りないときは small のクラスタで埋める`; `self.best ... hyps.insert(0, self.best)`
