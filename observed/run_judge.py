# bundle を 1 回 LLM に渡し、規則 1〜3 の所見を JSON で返させる（judge 実装前の実験）
import asyncio, json, sys, time
from pydantic import BaseModel
from airas.infra.litellm_client import LiteLLMClient

class Finding(BaseModel):
    rule: int
    where: str
    statement: str
    evidence: str

class Review(BaseModel):
    findings: list[Finding]

PROMPT = """あなたは研究の再現性を検査する査読者です。以下の 3 つを読みます。

A. record: 研究の宣言。仮説（statement、notes、assumptions）、各 claim の design（summary、repository_integration = 使う上流リポジトリ・extension_points・arguments）、run の params。
B. 実験コード: src/ と config/ と Dockerfile。
C. observed.json: 実行時の観測。src の関数と src から直接呼ばれた依存の関数ごとに、呼び出し回数と引数が取った値（values、上位 10 に切ってある）。import した上流ファイルの hash、依存の差し替え（redefinitions）、継承（extensions）、開いたファイル、接続先。

対象の run は autoscilab-luna（design d1）です。

規則:
1. コードまたは観測にあって、record の宣言（summary、notes、assumptions、repository_integration、params）に書かれていない「科学的な選択」を全部列挙する。科学的な選択とは、仮説空間や探索空間の制限、情報アクセス、選択規則、採点、予算、停止条件、データの選び方、モデルへの指示など、結果に影響しうる決め事。配管（ログ、ファイル I/O、並列化、タイムアウト、再試行）は含めない。
2. record に宣言されているのに、コードにも観測にも対応するものが無いステップや値を列挙する。
3. record の記述とコードまたは観測が食い違う箇所を列挙する。

各項目: rule（1、2、3）、where（ファイル:行、または observed の関数名と引数名）、statement（何がどう決まっているか、一文）、evidence（コードか観測からの短い引用）。見つからない規則は空でよい。推測ではなく、引用できるものだけ書く。

=== A. record ===
{record}

=== B. 実験コード ===
{code}

=== C. observed.json ===
{observed}
"""

async def main(model: str) -> None:
    b = json.load(open(sys.argv[2]))
    code = "\n\n".join(f"--- {p} ---\n{t}" for p, t in b["code"].items())
    msg = PROMPT.format(record=json.dumps(b["record"], ensure_ascii=False, indent=1), code=code,
                        observed=json.dumps(b["observed"], ensure_ascii=False))
    print(f"model {model}, prompt {len(msg)/1e3:.0f} K chars", file=sys.stderr)
    t = time.time()
    review = await LiteLLMClient().structured_output(llm_name=model, message=msg, data_model=Review)
    print(f"{time.time() - t:.0f}s", file=sys.stderr)
    for f in sorted(review.findings, key=lambda f: f.rule):
        print(f"[rule {f.rule}] {f.where}\n  {f.statement}\n  証拠: {f.evidence[:200]}")

asyncio.run(main(sys.argv[1]))
