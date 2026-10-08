"""SciGym のデータ: 無ければ HuggingFace（h4duan/scigym-sbml）から公式 data/download.py と同じ配置で書き出す。"""

from pathlib import Path


def download_split(split: str, root: Path) -> None:
    from datasets import load_dataset

    for row in load_dataset("h4duan/scigym-sbml", split=split):
        d = root / split / row["folder_name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "truth.xml").write_text(row["truth_xml"])
        (d / "partial.xml").write_text(row["partial"])
        (d / "truth.sedml").write_text(row["truth_sedml"])


def instances(data_root: str, split: str, mode: str) -> list[Path]:
    """件のディレクトリ一覧。sanity は 1 件、pilot は等間隔の 10 件、full は全件"""
    data_dir = Path(data_root) / split
    if not data_dir.is_dir():
        download_split(split, Path(data_root))
    found = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if mode == "sanity":
        return found[:1]
    if mode == "pilot":
        return found[:: max(len(found) // 10, 1)][:10]
    return found
