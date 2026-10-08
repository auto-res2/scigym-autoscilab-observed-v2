"""sitecustomize.py の動作確認。`python .airas/test_sitecustomize.py` で実行する。

偽の上流 package と依存と実験コード（src/）を一時ディレクトリに作り、フック付きで走らせ、
結合した observed.json を検査する。"""

import glob
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET = secrets.token_hex(8)  # 伏せられるべき値。実行ごとに作る

UPSTREAM = """
from pathlib import Path
from textwrap import dedent
_g = {}
exec("def generated():\\n    return 1", _g)
generated = _g["generated"]  # exec 由来（定義元 "<string>"）
from dataclasses import dataclass
from abc import abstractmethod
@dataclass
class Config:
    n: int = 1
def propose(data, n_basis=10, *, seed=None):
    return [1, 2, 3]
def stream():
    yield "a"
    yield "b"
def connect(url, api_key="x"):
    return url
import abc
class Controller(abc.ABC):  # ABC は _abc_impl を各クラスに置く。override に数えないこと
    def run(self, max_iterations, eval_debug_rounds=5):
        return list(stream()) + propose(None)   # 依存同士の呼び出し。記録しない
    def helper(self):
        return 0
class NotGiven:                 # openai 流の「省略」の印
    def __repr__(self):
        return "NOT_GIVEN"
NOT_GIVEN = NotGiven()
def create(model, n=NOT_GIVEN):
    return model
def inner(y):                   # 依存同士の呼び出し。記録しない
    return y
def helper_fn(x):
    return inner(x)
"""

FAKEDEP = """
class Client:
    pass
"""

EXPERIMENT = """
import os, socket, subprocess, sys, threading
import fakepkg, fakedep
class Replacement:                                            # 依存のクラスの差し替え（メソッド無し）
    pass
fakedep.Client = Replacement
class Tuned(fakepkg.Controller):                              # 継承と override
    def run(self, *a, **k):
        return super().run(*a, **k)
    def extra(self):
        return 0
def step(i):
    return i
def prompt(text):
    return len(text)
def gen():
    yield 1
def main():
    secret = os.environ["MY_SECRET_VALUE"]
    for i in range(60):                                       # 異なり値 50 の上限を超える
        step(i)
    for _ in range(3):                                        # 長い文字列は sha で数える
        prompt("p" * 300)
        prompt("q" * 400)
    list(gen())
    fakepkg.helper_fn({"Authorization": "sk-" + "a" * 30})    # 辞書の中の鍵も sha に
    fakepkg.helper_fn(7)
    fakepkg.helper_fn("sk-" + "a" * 30)                       # 鍵の形は sha に
    fakepkg.helper_fn(object())                               # アドレス入り repr は型だけ
    fakepkg.create("m")                                       # 省略の印の n は記録しない
    open(__file__).close()
    open(os.path.join(os.environ["AIRAS_OBSERVE_DIR"], "w.txt"), "w").close()
    open(os.path.join(os.environ["AIRAS_OBSERVE_DIR"], "w.txt"), "a").close()
    socket.getaddrinfo("localhost", 80)
    fakepkg.Controller().run(20)
    fakepkg.propose([0] * 1000, seed=3)                       # 大きいコンテナは型だけ
    t = threading.Thread(target=lambda: fakepkg.propose("thread")); t.start(); t.join()
    fakepkg.connect("http://h:8000/v1", api_key=secret)          # 値で伏せる
    fakepkg.connect(f"http://h:8000/v1?k={secret}", api_key="short")  # URL に含まれても伏せる
    fakepkg.propose({"headers": {"Authorization": f"Bearer {secret}"}})  # dict の repr でも
    fakepkg.propose = lambda *a, **k: []                  # 関数の差し替え
    fakepkg.Controller.helper = lambda self: 1            # メソッドの差し替え
    fakepkg.Controller.ext = staticmethod(fakepkg.dedent) # 外部定義の関数を載せる。記録しない
    fakepkg.Path.is_dir = lambda self: True               # import したクラスのメソッドの差し替え
    os.putenv("FOO", "1")
    subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); import adapter; adapter.step(99)"], check=True)
    subprocess.run([sys.executable, "-IS", "-c", "print(1)"], check=True, capture_output=True)
    with open(__file__, "a") as f:                        # import 後の書き換えは hash に出ない
        f.write("# edited after import\\n")
    sys.setprofile(None)                                  # フックを外す
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("fakepkg", "fakedep", "src", "out"):
            os.makedirs(f"{tmp}/{d}")
        with open(f"{tmp}/fakepkg/__init__.py", "w") as f:
            f.write(textwrap.dedent(UPSTREAM))
        with open(f"{tmp}/fakedep/__init__.py", "w") as f:
            f.write(textwrap.dedent(FAKEDEP))
        with open(f"{tmp}/src/adapter.py", "w") as f:
            f.write(textwrap.dedent(EXPERIMENT))
        with open(f"{tmp}/run.py", "w") as f:
            f.write(
                "import sys; sys.path.insert(0, 'src'); import adapter; adapter.main()\n"
            )
        env = {
            **os.environ,
            "PYTHONPATH": HERE,
            "AIRAS_OBSERVE_DIR": f"{tmp}/out",
            "AIRAS_SECRET_NAMES": "MY_SECRET_VALUE",  # 基盤が渡す名前一覧
            "MY_SECRET_VALUE": SECRET,
            "FAKE_TOKEN": "t0kenvalue2",  # 一覧に無くても名前の規則で伏せる
            "FAKE_MODE": "fast",
        }
        subprocess.run([sys.executable, "run.py"], cwd=tmp, env=env, check=True)
        files = sorted(glob.glob(f"{tmp}/out/*.json"))
        assert len(files) == 2, files  # 親と、フック付きの子（-IS の子は書かない）
        recs = [json.load(open(f)) for f in files]
        parent = next(r for r in recs if r["process"]["argv"] == ["run.py"])
        child = next(r for r in recs if r["process"]["argv"] == ["-c"])
        assert parent["errors"] == [], parent["errors"]
        assert child["calls"]["adapter.step"]["calls"] == 1
        assert list(parent["hook"]) == ["sha256"]

        # 結合: 定義は先に始まった親が勝ち、calls と reaches は回数を足す
        subprocess.run(
            [sys.executable, f"{HERE}/sitecustomize.py", "merge", f"{tmp}/out", "t", f"{tmp}/observed.json"],
            check=True,
        )
        merged = json.load(open(f"{tmp}/observed.json"))
        assert merged["version"] == 2 and merged["run_id"] == "t"
        assert merged["errors"] == [], merged["errors"]
        assert len(merged["processes"]) == 2
        # env は子に FOO が足されているので同じにならず、各プロセスに残る
        assert "env" not in merged and all("env" in p for p in merged["processes"])
        env_rec = next(p["env"] for p in merged["processes"] if p["argv"] == ["run.py"])
        assert env_rec["FAKE_MODE"] == "fast"
        assert env_rec["AIRAS_SECRET_NAMES"] == "MY_SECRET_VALUE"  # 名前の一覧は伏せない
        assert env_rec["MY_SECRET_VALUE"]["redacted"] == "MY_SECRET_VALUE"
        assert env_rec["FAKE_TOKEN"]["redacted"] == "FAKE_TOKEN"

        calls = merged["calls"]
        run = calls["fakepkg.Controller.run"]  # 実験コードから直接呼んだ依存
        assert run["calls"] == 1
        assert run["args"]["max_iterations"] == {
            "type": "int",
            "calls": 1,
            "distinct": 1,
            "values": [{"value": 20, "calls": 1}],
            "min": 20,
            "max": 20,
        }
        assert run["args"]["eval_debug_rounds"]["values"] == [{"value": 5, "calls": 1}]
        assert run["samples"][0]["ret"]["type"] == "list"
        assert "fakepkg.stream" not in calls and "fakepkg.inner" not in calls  # 依存同士の呼び出し
        propose = calls["fakepkg.propose"]
        assert propose["calls"] == 3  # 実験コードから 3 回。run() の中の 1 回と差し替え後の lambda は数えない
        assert propose["args"]["seed"]["values"] == [
            {"value": None, "calls": 2},
            {"value": 3, "calls": 1},
        ]
        data = {json.dumps(v["value"], sort_keys=True) for v in propose["args"]["data"]["values"]}
        assert json.dumps("thread") in data  # 平文
        assert json.dumps({"type": "list"}, sort_keys=True) in data  # 大きいコンテナは型だけ
        assert any('"redacted": "MY_SECRET_VALUE"' in d for d in data)  # 小さい dict の repr の中の秘密
        assert propose["args"]["data"]["length_max"] == 1000
        connect = calls["fakepkg.connect"]
        assert connect["calls"] == 2
        assert {json.dumps(v["value"]) for v in connect["args"]["api_key"]["values"]} == {
            json.dumps({"redacted": "MY_SECRET_VALUE", "len": len(SECRET)}),
            json.dumps("short"),
        }
        assert any(
            isinstance(v["value"], dict) and v["value"].get("redacted") == "MY_SECRET_VALUE"
            for v in connect["args"]["url"]["values"]
        )  # URL に含まれても伏せる
        helper = calls["fakepkg.helper_fn"]
        assert helper["calls"] == 4
        kinds = {json.dumps(v["value"], sort_keys=True) for v in helper["args"]["x"]["values"]}
        assert json.dumps(7) in kinds
        assert any('"sha256"' in k and '"len": 33' in k for k in kinds)  # 鍵の形は sha
        assert any('"type": "dict"' in k and '"sha256"' in k for k in kinds)  # 辞書の中の鍵も
        assert json.dumps({"type": "object"}, sort_keys=True) in kinds  # アドレスは残さない
        assert list(calls["fakepkg.create"]["args"]) == ["model"]  # 省略の印は値ではない
        step = calls["adapter.step"]
        assert step["calls"] == 61  # 親 60 回 + 子 1 回
        assert step["args"]["i"]["distinct"] == 61 and len(step["args"]["i"]["values"]) == 50
        assert (step["args"]["i"]["min"], step["args"]["i"]["max"]) == (0, 99)
        text = calls["adapter.prompt"]["args"]["text"]
        assert text["distinct"] == 2 and text["calls"] == 6
        assert (text["length_min"], text["length_max"]) == (300, 400)
        assert all("sha256" in v["value"] for v in text["values"])
        gen = calls["adapter.gen"]
        assert gen["calls"] == 1 and "ret" not in gen["samples"][0]  # 再開は数えず、戻り値も取らない
        assert calls["adapter.main"]["calls"] == 1
        assert not any("<lambda>" in fn for fn in calls)
        assert "adapter.Tuned" not in calls  # クラス本体の実行は呼び出しではない
        dumped = json.dumps(merged)
        assert SECRET not in dumped and "t0kenvalue2" not in dumped and "sk-" + "a" * 30 not in dumped

        assert merged["src_modules"] == {
            "src/adapter.py": hashlib.sha256(textwrap.dedent(EXPERIMENT).encode()).hexdigest()
        }  # import した時の内容。その後の追記は含まない
        hashes = merged["loaded_file_hashes"]
        assert hashes["fakepkg"]["sha256"] and hashes["fakedep"]["sha256"]
        assert "os" not in hashes and "pathlib" not in hashes  # stdlib は hash しない
        assert merged["redefinitions"] == {
            "fakepkg.generated": "<string>",  # exec 由来は出自不明として残す
            "fakepkg.propose": "src/adapter.py",
            "fakepkg.Controller.helper": "src/adapter.py",
            "pathlib.Path.is_dir": "src/adapter.py",  # import したクラスへの差し替えは定義元のモジュールで見る
            "fakedep.Client": "src/adapter.py",  # クラスごとの差し替え
        }  # Config の生成 dunder、外部定義の Controller.ext、import しただけの名前は含まない
        assert merged["extensions"] == {
            "adapter.Tuned": {"bases": ["fakepkg.Controller"], "overrides": ["run"]}
        }  # abc.ABC は stdlib なので基底に数えない

        r = merged["reaches"]
        assert r["opens"]["src/adapter.py"]["r"] == 1  # cwd からの相対パス
        assert (r["opens"]["out/w.txt"]["w"], r["opens"]["out/w.txt"]["a"]) == (1, 1)
        assert all(v["experiment_code"].startswith(f"{tmp}/src/") for v in r["opens"].values())
        assert r["getaddrinfo"] == {"localhost": 1}
        assert [(s["hooked"], s["n"]) for s in r["spawns"]] == [(True, 1), (False, 1)]
        assert r["env_changes"] == {"FOO": 1}
        assert [t["event"] for t in r["tamper"]] == ["sys.setprofile"]
        assert r["tamper"][0]["experiment_code"].startswith(f"{tmp}/src/")
    print("ok")


if __name__ == "__main__":
    main()
