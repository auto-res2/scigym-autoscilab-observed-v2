"""`make run` が起動した Python プロセスの実行記録（version 3）。

Makefile が PYTHONPATH にこのディレクトリを足すので、Python はどのコードより先に
このファイルを import する。AIRAS_OBSERVE_DIR が無ければ何もしない。終了時に
AIRAS_OBSERVE_DIR/<pid>-<開始時刻>.json へ書き、Makefile が `merge` で observed.json に結合する。
record の宣言は読まない。観測の範囲が宣言で狭まらないためで、何を宣言と比べるかは gate が決める。

- calls: 関数ごとに 1 項目。対象は実験コード（src/）で定義された関数と、実験コードから直接
  呼ばれた依存の関数（stdlib、依存同士の呼び出し、`_` で始まる内部名は除く。`__init__` と `__call__` は見る）。項目は呼び出し回数、引数ごと
  （`args` はリスト。観測された引数名は鍵ではなく `name` に置き、辞書の鍵は全部この記録の語彙にする）の
  「取った値 → 回数」（回数の多い 50 値。異なり数は 1000 まで数え、そこまでは回数も正確。
  数値は min/max、長さのあるものは length_min/max）、先頭 3 回の全引数と戻り値。
  観測した値は 1 件ずつ辞書で書く。`value` があれば本物の値（数・文字列・JSON にできる要素 20 個以下の
  コンテナ。200 文字まで）。無ければ中身は保存していない: `truncated: true` と型・長さ（文字列と小さい
  コンテナは sha256 も）、または秘密なら `redacted: <環境変数名>` と長さ。鍵の形（sk- / ghp_ / hf_ /
  AKIA / JWT …）の文字列も保存しない。秘密の値は、基盤が AIRAS_SECRET_NAMES で渡す名前（Actions secrets の
  一覧。ローカルでは ~/.airas/credentials.json のキー）の環境変数から集める。大きい配列やオブジェクトは
  値の一覧に入らず、引数の type と length_min/max だけ残る
- src_modules: 実験コードの各ファイルの sha256。そのコードが初めて走った時（import 直後）に
  読むので、後からの書き換えは入らない（.pyc は見ない）。gate が実行コミットの同じファイルと比べる
- loaded_file_hashes: uv.lock が守らないモジュールのファイルの sha256。守られているのは、lock に
  index（registry）由来として同じ版で載っている配布物だけ。git / URL / ローカル由来、lock に無い
  追加 install、PYTHONPATH に乗せた clone は全部 hash する。gate が record のリポジトリの
  スナップショットと比べ、上流が原本のまま走ったかを見る
- redefinitions: 依存（上流を含む）の名前空間にある名前のうち、定義元が実験コードのもの。
  monkeypatch とクラスの差し替え
- extensions: 実験コードのクラスのうち stdlib 以外のクラスを継承するもの。基底と override したメソッド名
- reaches: 実験コードが起点の open（インタプリタと依存の配下は除く。一時ディレクトリはディレクトリに
  畳む）、connect、名前解決、実験コードが起動した（または python の）子プロセス、実験コードによる
  環境変数の変更、実験コードが直接呼んだ exec / eval、このフックを外す操作。回数で集約
- process: argv、Python 版、起動時の環境変数（name と、引数と同じ規則の値）
"""

import atexit
import hashlib
import json
import os
import re
import sys
import sysconfig
import tempfile
import threading
import time
import types

_OUT_DIR = os.environ.get("AIRAS_OBSERVE_DIR")
_SELF = os.path.abspath(__file__)
_CWD = os.getcwd()
_EXPERIMENT_CODE = os.path.join(_CWD, "src") + os.sep
_STDLIB = tuple(
    {sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["platstdlib"]}
)
# 実験コード起点でも記録しない open: インタプリタと依存の配下、擬似ファイルシステム、OS のデータ
_LIBRARY = tuple(
    p.rstrip(os.sep) + os.sep
    for p in {sys.base_prefix, sys.prefix, *_STDLIB, "/proc", "/sys", "/dev", "/usr/share"}
)
_TMP = tempfile.gettempdir()
_GENERATOR = 0x20 | 0x80 | 0x200  # CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR
_SECRET_NAME = re.compile(
    r"key|token|secret|passw|credential|auth|private|cookie|session|header",
    re.IGNORECASE,
)
# ponytail: 既知の鍵の接頭辞だけ。新しいプロバイダが出たら足す
_KEY_LIKE = re.compile(r"(sk-|ghp_|gho_|github_pat_|hf_|AKIA|eyJ|xox[abp]-|AIza|glpat-)\S{10,}")
_OMITTED = ("NotGiven", "NotGivenType", "Sentinel", "_NoValueType")  # 省略の印は値ではない
_SAMPLES, _VALUES, _DISTINCT, _SMALL = 3, 50, 1000, 20


def _secret_names() -> set[str]:
    """伏せる環境変数の名前。基盤が渡す AIRAS_SECRET_NAMES（Actions secrets の名前一覧）、
    無ければローカルの ~/.airas/credentials.json のキー。名前の規則は足し忘れの保険"""
    names = {n for n in os.environ.get("AIRAS_SECRET_NAMES", "").split(",") if n}
    if not names:
        try:
            with open(os.path.expanduser("~/.airas/credentials.json")) as f:
                names = set(json.load(f))
        except (OSError, ValueError):
            pass
    # AIRAS_SECRET_NAMES は名前の一覧であって値ではない
    return (names | {n for n in os.environ if _SECRET_NAME.search(n)}) - {
        "AIRAS_SECRET_NAMES"
    }


_SECRET_NAMES = _secret_names()
# 伏せる値 → 名前。8 文字未満は誤爆するので対象外
_SECRET_VALUES = {
    os.environ[n]: n for n in _SECRET_NAMES if len(os.environ.get(n, "")) >= 8
}

_kind: dict[types.CodeType, str] = {}  # code → "src" | "dep" | ""（見ない）
_src_hashes: dict[str, str | None] = {}  # 実験コードの相対パス → import 時の sha256
_first_lasti: dict[types.CodeType, int] = {}
_active: dict[int, dict] = {}  # 戻り値を待つ sample
_fns: dict[str, dict] = {}
_opens: dict[str, dict] = {}
_connects: dict[str, int] = {}
_lookups: dict[str, int] = {}
_spawns: dict[str, dict] = {}
_env_changes: dict[str, int] = {}
_execs: dict[str, int] = {}
_tamper: list[dict] = []
_errors: list[str] = []
_started = time.time()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return _sha(f.read())
    except OSError:
        return None


def _is_stdlib(file: str) -> bool:
    return (file.startswith(_STDLIB) or file.startswith("<")) and (
        "site-packages" not in file and "dist-packages" not in file
    )


def _relative(file: str) -> str:
    """実験コードは cwd からの相対パスで。gate は "src/" で見る"""
    return os.path.relpath(file, _CWD) if file.startswith(_CWD + os.sep) else file


def _json_safe(v) -> bool:
    """JSON にして元に戻せるか。文字列でない辞書の鍵は json.dumps が文字列に潰すので除く"""
    if isinstance(v, dict):
        return all(isinstance(k, str) and _json_safe(x) for k, x in v.items())
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(_json_safe(x) for x in v)
    return True


def _item(v, name: str = "") -> dict:
    """観測した値 1 件の記録。value があれば本物の値、無ければ中身は保存していない（docstring 参照）。
    name は引数名か環境変数名。中身を保存しない大きいものは repr も取らない（ホットループで重い）"""
    if v is None or isinstance(v, (bool, int, float)):
        return {"value": v}
    text, value = None, None
    if isinstance(v, str):
        text = value = v
    elif isinstance(v, (list, tuple, set, frozenset, dict)) and len(v) <= _SMALL and _json_safe(v):
        try:
            value = sorted(v, key=repr) if isinstance(v, (set, frozenset)) else v
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)  # gate と同じ正規形で hash する
            value = json.loads(text)
        except (TypeError, ValueError):
            text = value = None
    if text is not None:
        secret = name if name in _SECRET_NAMES else None
        if secret is None:
            # コンテナの中の秘密は JSON でエスケープされているので、その形でも探す
            secret = next(
                (
                    n
                    for sv, n in _SECRET_VALUES.items()
                    if sv in text or json.dumps(sv, ensure_ascii=False)[1:-1] in text
                ),
                None,
            )
        if secret is not None:
            return {"redacted": secret, "len": len(text)}
        if len(text) <= 200 and not _KEY_LIKE.search(text):
            return {"value": value}
    item: dict = {"truncated": True, "type": type(v).__name__}
    if hasattr(v, "__len__"):
        try:
            item["len"] = len(v)
        except Exception:
            pass
    if text is not None:
        item["sha256"] = _sha(text.encode())
    return item


def _where():
    """イベントを起こした Python の場所（caller）、その上にある実験コードの場所、そして
    stdlib を抜けて最初に出会うのが実験コードか（実験コードが直接起こしたか）。
    このファイルの hook 関数の分だけ上に辿る。起こしたのがこのファイル自身なら "self"。"""
    f = sys._getframe(0)
    while f is not None and f.f_code in _HOOK_CODES:
        f = f.f_back
    if f is None:
        return None, None, False
    if f.f_code.co_filename == _SELF:
        return "self", None, False
    caller = f"{f.f_code.co_filename}:{f.f_lineno}"
    direct = None
    while f is not None:
        file = f.f_code.co_filename
        if direct is None and not _is_stdlib(file):
            direct = file.startswith(_EXPERIMENT_CODE)
        if file.startswith(_EXPERIMENT_CODE):
            return caller, f"{file}:{f.f_lineno}", bool(direct)
        f = f.f_back
    return caller, None, False


def _classify(code) -> str:
    if code.co_name.startswith("<") or not code.co_flags & 0x02:  # lambda、内包表記、クラス本体
        return ""
    file = code.co_filename
    if file.startswith(_EXPERIMENT_CODE):
        return "src"
    if _is_stdlib(file) or file == _SELF:
        return ""
    if code.co_name.startswith("_") and code.co_name not in ("__init__", "__call__"):
        return ""  # 依存の内部名（numpy の _dispatcher など）
    return "dep"


def _note(a: dict, v, name: str) -> dict:
    """引数 1 つの集計。記録した item を返す"""
    a["calls"] += 1
    t = type(v).__name__
    a["types"][t] = a["types"].get(t, 0) + 1
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        a["min"] = v if "min" not in a else min(a["min"], v)
        a["max"] = v if "max" not in a else max(a["max"], v)
    elif hasattr(v, "__len__"):
        try:
            n = len(v)
            a["length_min"] = n if "length_min" not in a else min(a["length_min"], n)
            a["length_max"] = n if "length_max" not in a else max(a["length_max"], n)
        except Exception:
            pass
    item = _item(v, name)
    if "value" in item or "sha256" in item or "redacted" in item:  # 型だけの item は数えない
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in a["values"]:
            a["values"][key]["calls"] += 1
        elif len(a["values"]) < _DISTINCT:
            a["values"][key] = {**item, "calls": 1}
    return item


def _profile(frame, event, arg):
    # 関数の call / return を受け、対象なら引数を関数ごとに集計し、先頭 3 回は戻り値も取る
    if event[1] == "_":  # c_call / c_return / c_exception は見ない
        return
    try:
        code = frame.f_code
        if event == "call":
            kind = _kind.get(code)
            if kind is None:
                file = code.co_filename
                if file.startswith(_EXPERIMENT_CODE):  # 初見 = import 直後。今の内容が走った内容
                    _src_hashes.setdefault(_relative(file), _file_sha(file))
                kind = _kind[code] = _classify(code)
            if not kind:
                return
            if kind == "dep":  # 依存は実験コードから直接呼ばれたときだけ
                back = frame.f_back
                if back is None or not back.f_code.co_filename.startswith(_EXPERIMENT_CODE):
                    return
            generator = code.co_flags & _GENERATOR
            if generator:
                # ジェネレータは再開のたびに call が来る。最小の f_lasti が初回の入口
                first = _first_lasti.get(code)
                if first is None or frame.f_lasti < first:
                    _first_lasti[code] = first = frame.f_lasti
                if frame.f_lasti > first:
                    return
            name = f"{frame.f_globals.get('__name__', '')}.{getattr(code, 'co_qualname', code.co_name)}"
            fn = _fns.get(name)
            if fn is None:
                fn = _fns[name] = {"calls": 0, "args": {}, "samples": []}
            fn["calls"] += 1
            sampling = len(fn["samples"]) < _SAMPLES
            n = code.co_argcount + code.co_kwonlyargcount
            names = list(code.co_varnames[:n])
            if code.co_flags & 0x04:
                names.append(code.co_varnames[n])
                n += 1
            if code.co_flags & 0x08:
                names.append(code.co_varnames[n])
            loc = frame.f_locals
            sample: list = []
            for k in names:
                if k not in loc or k == "self":
                    continue
                v = loc[k]
                if type(v).__name__ in _OMITTED:
                    continue
                a = fn["args"].get(k)
                if a is None:
                    a = fn["args"][k] = {"calls": 0, "types": {}, "values": {}}
                item = _note(a, v, k)
                if sampling:
                    sample.append({"name": k, **item})
            if sampling:
                rec = {"args": sample}
                fn["samples"].append(rec)
                if not generator:  # yield でも return が来るので戻り値は取らない
                    _active[id(frame)] = rec
        elif event == "return":
            rec = _active.pop(id(frame), None)
            if rec is not None:
                rec["ret"] = _item(arg)
    except Exception as e:  # 観測の不具合で run を止めない
        if len(_errors) < 100:
            _errors.append(f"profile {event}: {e!r}")


def _audit(event, args):
    # audit イベントを受け、_where() で発生源を特定して reaches に積む
    try:
        if event == "open":
            caller, code, _ = _where()
            # 実験コードが起点の open だけ。import 時、fd、ライブラリ配下は依存の内部なので見ない
            if (
                code is None
                or isinstance(args[0], int)
                or (caller or "").startswith("<frozen importlib")
            ):
                return
            path = os.path.abspath(str(args[0]))
            if path.startswith(_LIBRARY):
                return
            if path.startswith(_CWD + os.sep):
                path = os.path.relpath(path, _CWD)
            elif path.startswith(_TMP + os.sep):
                path = _TMP  # 一時ファイルは名前がランダムなのでディレクトリに畳む
            rec = _opens.setdefault(path, {"experiment_code": code})
            mode = str(args[1])
            rec[mode] = rec.get(mode, 0) + 1
        elif event == "socket.connect":
            address = args[1]
            key = f"{address[0]}:{address[1]}" if isinstance(address, tuple) else str(address)
            _connects[key] = _connects.get(key, 0) + 1
        elif event == "socket.getaddrinfo":
            host = args[0].decode() if isinstance(args[0], bytes) else str(args[0])
            _lookups[host] = _lookups.get(host, 0) + 1
        elif event in ("subprocess.Popen", "os.exec", "os.posix_spawn"):
            argv = [str(a) for a in (args[1] or [])]
            env = args[3] if event == "subprocess.Popen" else args[2]
            env = os.environ if env is None else env
            # 子にもこのフックが入るか: PYTHONPATH を引き継ぎ、-I/-S/-E で site を切っていない
            hooked = (
                os.path.dirname(_SELF) in str(env.get("PYTHONPATH", ""))
                and "AIRAS_OBSERVE_DIR" in env
            )
            python = bool(argv) and "python" in os.path.basename(argv[0])
            if hooked and python:
                for a in argv[1:]:
                    if not a.startswith("-"):
                        break
                    if not a.startswith("--") and set(a[1:]) & {"I", "S", "E"}:
                        hooked = False
                    if a[:2] in ("-c", "-m"):
                        break
            caller, code, direct = _where()
            if direct or python:  # 依存が起動する lscpu などは見ない
                key = json.dumps([event, argv[:50], hooked])
                rec = _spawns.get(key)
                if rec is None:
                    rec = _spawns[key] = {
                        "event": event,
                        "argv": argv[:50],
                        "hooked": hooked,
                        "experiment_code": code,
                        "n": 0,
                    }
                rec["n"] += 1
            if event == "os.exec":  # 成功すると atexit が走らないので今書く
                _finish()
        elif event == "compile" and str(args[1]).startswith("<"):
            # 文字列からの exec / eval / compile（ファイル名が "<string>" など）。ファイルの import は見ない。
            # 実験コードの行が直接呼んだものだけ。dataclass が生成する __init__ は stdlib が呼ぶので入らない
            caller, code, _ = _where()
            if code and (caller or "").startswith(_EXPERIMENT_CODE):
                _execs[code] = _execs.get(code, 0) + 1
        elif event in ("os.putenv", "os.unsetenv"):
            caller, code, direct = _where()
            if direct:  # 依存が自分の都合で触る OPENBLAS_* などは見ない
                name = os.fsdecode(args[0])
                _env_changes[name] = _env_changes.get(name, 0) + 1
        elif event in ("sys.setprofile", "sys.settrace", "sys.addaudithook"):
            caller, code, direct = _where()
            if caller == "self" or (caller and os.sep + "threading.py:" in caller):
                return
            if event == "sys.addaudithook" and not direct:  # filelock などの監査は外しではない
                return
            _tamper.append({"event": event, "caller": caller, "experiment_code": code})
    except Exception as e:  # 観測の不具合で run を止めない
        if len(_errors) < 100:
            _errors.append(f"audit {event}: {e!r}")


_HOOK_CODES = {_where.__code__, _profile.__code__, _audit.__code__}


def _reset_after_fork():
    for c in (_fns, _active, _opens, _connects, _lookups, _spawns, _env_changes, _execs):
        c.clear()
    _tamper.clear()
    _errors.clear()


def _from_experiment(file: str) -> bool:
    return file.startswith(_EXPERIMENT_CODE)


def _locked_modules() -> set[str]:
    """uv.lock の hash が守る最上位モジュール名: index（registry）由来として lock に載り、
    入っている版も同じ配布物のもの。lock が無ければ空（= 全部 hash する）"""
    import importlib.metadata as metadata
    import tomllib

    def norm(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    found: set[str] = set()
    try:
        with open(os.path.join(_CWD, "uv.lock"), "rb") as f:
            lock = tomllib.load(f)
        locked = {
            (norm(pkg["name"]), pkg.get("version"))
            for pkg in lock.get("package", [])
            if "registry" in pkg.get("source", {})
        }
        for module, dists in metadata.packages_distributions().items():
            if any((norm(d), metadata.version(d)) in locked for d in dists):
                found.add(module)
    except Exception as e:  # lock や metadata が読めなければ守られていない扱い
        if len(_errors) < 100:
            _errors.append(f"lock: {e!r}")
    return found


def _dependency_file(cls) -> str | None:
    """stdlib でも実験コードでもないモジュールで定義されたクラスなら、そのモジュールのファイル"""
    file = getattr(sys.modules.get(cls.__module__), "__file__", None)
    if file and not _is_stdlib(file) and not file.startswith(_EXPERIMENT_CODE):
        return file
    return None


def _definitions():
    """(loaded_file_hashes, redefinitions)。実験コード以外の全モジュールを見る"""
    hashes, redefined = {}, {}
    locked = _locked_modules()
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", None)
        if not file or file.startswith(_EXPERIMENT_CODE) or file == _SELF:
            continue
        if not _is_stdlib(file) and name.split(".")[0] not in locked:
            hashes[name] = {"file": file, "sha256": _file_sha(file)}
        for attr, obj in list(vars(mod).items()):
            if attr.startswith("__"):
                continue
            try:
                if isinstance(obj, types.FunctionType):
                    if _from_experiment(obj.__code__.co_filename):
                        redefined[f"{name}.{attr}"] = _relative(obj.__code__.co_filename)
                elif isinstance(obj, type):
                    defined_in = getattr(sys.modules.get(obj.__module__), "__file__", None) or ""
                    if defined_in.startswith(_EXPERIMENT_CODE):  # 依存のクラスを実験コードのクラスで差し替え
                        redefined[f"{name}.{attr}"] = _relative(defined_in)
                    elif obj.__module__ == name:  # import したクラスは定義元のモジュールで見る
                        for member, value in list(vars(obj).items()):
                            if isinstance(value, (staticmethod, classmethod)):
                                value = value.__func__
                            if not isinstance(value, types.FunctionType) or member.startswith("__"):
                                continue  # dataclass 等が生成する dunder は数えない
                            if _from_experiment(value.__code__.co_filename):
                                redefined[f"{name}.{attr}.{member}"] = _relative(value.__code__.co_filename)
            except Exception:  # 触ると import を試みる遅延 proxy などは飛ばす
                continue
    return hashes, redefined


def _extensions() -> dict:
    """実験コードで定義されたクラスのうち、stdlib 以外のクラスを継承するもの"""
    found = {}
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", None)
        if not (file and file.startswith(_EXPERIMENT_CODE)) or name == "__mp_main__":
            continue  # __mp_main__ は multiprocessing が読み直した __main__ の写し
        for attr, obj in list(vars(mod).items()):
            if not (isinstance(obj, type) and obj.__module__ == name):
                continue
            bases = [b for b in obj.__mro__[1:] if _dependency_file(b)]
            if bases:
                found[f"{name}.{attr}"] = {
                    "bases": [f"{b.__module__}.{b.__qualname__}" for b in bases],
                    # 基底にもある関数メンバーだけ。ABC が置く _abc_impl などの属性は数えない
                    "overrides": [
                        m
                        for m, v in vars(obj).items()
                        if not m.startswith("__")
                        and isinstance(v, (types.FunctionType, staticmethod, classmethod))
                        and any(m in vars(b) for b in bases)
                    ],
                }
    return found


def _finish():
    hashes, redefined = _definitions()
    out = {
        "version": 3,
        "hook": {"sha256": _file_sha(_SELF)},  # 誰が観察したか
        "process": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "argv": sys.argv,
            "cwd": _CWD,
            "python": sys.version.split()[0],
            "env": [{"name": k, **_item(v, k)} for k, v in sorted(os.environ.items())],
            "started": _started,
            "ended": time.time(),
        },
        "src_modules": dict(_src_hashes),
        "loaded_file_hashes": hashes,
        "redefinitions": redefined,
        "extensions": _extensions(),
        "calls": _fns,
        "reaches": {
            "opens": _opens,
            "connects": _connects,
            "getaddrinfo": _lookups,
            "spawns": _spawns,
            "env_changes": _env_changes,
            "execs": _execs,
            "tamper": _tamper,
        },
        "errors": _errors,
    }
    path = os.path.join(_OUT_DIR, f"{os.getpid()}-{int(_started * 1000)}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, default=str, indent=1)


def install() -> None:
    """import 時（Python が sitecustomize として読んだとき）: フックを入れる"""
    os.makedirs(_OUT_DIR, exist_ok=True)
    sys.addaudithook(_audit)
    sys.setprofile(_profile)
    threading.setprofile(_profile)
    os.register_at_fork(after_in_child=_reset_after_fork)
    atexit.register(_finish)


def _add(dst: dict, src: dict) -> None:
    """記録を足す。回数は和、min/max はその通り、辞書は再帰、samples と tamper は連結、他は先勝ち"""
    for k, v in src.items():
        if k not in dst or k in ("value", "len"):
            dst.setdefault(k, v)
        elif isinstance(v, dict):
            _add(dst[k], v)
        elif isinstance(v, list):
            if k in ("samples", "tamper"):
                dst[k] = dst[k] + v
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if k in ("min", "length_min"):
                dst[k] = min(dst[k], v)
            elif k in ("max", "length_max"):
                dst[k] = max(dst[k], v)
            else:
                dst[k] += v


def merge(d: str, run_id: str, out: str) -> None:
    """プロセスごとの記録を observed.json に結合する。定義は先に始まったプロセス（親。
    差し替え後の姿）が勝ち、calls と reaches は回数を足す"""
    import glob

    processes = sorted(
        (json.load(open(f)) for f in glob.glob(d + "/*.json")),
        key=lambda p: p["process"]["started"],
    )
    merged: dict = {"version": 3, "run_id": run_id}
    for p in processes:
        p.pop("version", None)
        process = p.pop("process")
        errors = p.pop("errors", [])
        for key, section in p.items():
            _add(merged.setdefault(key, {}), section)
        merged.setdefault("errors", []).extend(errors)
        p["process"] = process
    envs = [p["process"]["env"] for p in processes]
    if envs and all(e == envs[0] for e in envs):
        merged["env"] = envs[0]
        for p in processes:
            p["process"].pop("env")
    for fn in merged.get("calls", {}).values():
        fn["samples"] = fn["samples"][:_SAMPLES]
        for a in fn["args"].values():
            a["type"] = "|".join(sorted(a.pop("types")))
            values = a.pop("values")
            if values:  # 型だけの引数（大きい配列やオブジェクト）には値の一覧が無い
                a["distinct"] = min(len(values), _DISTINCT)
                a["values"] = sorted(values.values(), key=lambda e: -e["calls"])[:_VALUES]
        # 観測された引数名を鍵にしない: 鍵はこの記録の語彙だけ、名前は name に
        fn["args"] = [{"name": name, **a} for name, a in fn["args"].items()]
    if "spawns" in merged.get("reaches", {}):
        merged["reaches"]["spawns"] = list(merged["reaches"]["spawns"].values())
    merged["processes"] = [p["process"] for p in processes]
    with open(out, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":  # python3 sitecustomize.py merge <dir> <run_id> <out>
    merge(*sys.argv[2:])
elif _OUT_DIR:
    install()
