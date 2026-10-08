# judge 実験の入力束を作る: record の宣言部分 / 実験コード / observed.json（values を上位 10 に切る）
import json, pathlib, sys
S = pathlib.Path(sys.argv[1])
rec = json.load(open(S / "dl/record.json"))
record = {
    "repositories": [{k: r.get(k) for k in ("id", "url", "commit", "method_entry")} for s in rec["literature"] for r in s.get("repositories", [])],
    "hypotheses": rec["hypotheses"],
}
code = {}
repo = S / "dl/scigym-autoscilab"
for p in sorted(list((repo / "src").glob("*.py")) + [repo / "config/config.yaml", repo / "config/run/autoscilab-luna.yaml", repo / "Dockerfile"]):
    code[str(p.relative_to(repo))] = p.read_text()
obs = json.load(open(next((S / "dl/run-v2b").rglob("observed.json"))))
for fn in obs["calls"].values():
    for a in fn["args"].values():
        a["values"] = a["values"][:10]
obs["processes"] = [{k: v for k, v in p.items() if k != "env"} for p in obs["processes"]]
out = S / "judge/bundle.json"
json.dump({"record": record, "code": code, "observed": obs}, open(out, "w"), ensure_ascii=False)
print({k: round(len(json.dumps(v, ensure_ascii=False)) / 1e3) for k, v in {"record": record, "code": code, "observed": obs}.items()}, "KB")
