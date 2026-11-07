"""Embed the current bp_protocol.py into the Colab notebooks."""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = base64.b64encode((ROOT / "src" / "bp_protocol.py").read_bytes()).decode("ascii")
MARKER = "base64.b64decode(\\\""
NOTEBOOKS = [
    ROOT / "notebooks" / "04_return_visit_bp.ipynb",
    ROOT / "notebooks" / "05_ambulatory_24h.ipynb",
    ROOT / "notebooks" / "06_ambulatory_24h_subsets.ipynb",
    ROOT / "notebooks" / "07_return_posture.ipynb",
    ROOT / "notebooks" / "03b_data_provenance.ipynb",
]


def replace_payload(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.find(MARKER)
    if start < 0:
        raise SystemExit(f"marker missing in {path}")
    start += len(MARKER)
    end = text.find("\\\")", start)
    if end < 0:
        raise SystemExit(f"payload end missing in {path}")
    path.write_text(text[:start] + PAYLOAD + text[end:], encoding="utf-8")
    print(f"embedded {path.name} ({len(PAYLOAD)} chars)")


def patch_notebook_05() -> None:
    path = ROOT / "notebooks" / "05_ambulatory_24h.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "run_ambulatory_24h(features, output_path" not in source:
            continue
        source = source.replace(
            "features = pd.read_csv(data_dir / 'features.tsv', sep='\\t')\n",
            "features = pd.read_csv(data_dir / 'features.tsv', sep='\\t')\n"
            "participants = pd.read_csv(data_dir / 'participants.tsv', sep='\\t')\n",
        )
        source = source.replace(
            "run_ambulatory_24h(features, output_path=str(output))",
            "run_ambulatory_24h(features, participants, output_path=str(output))",
        )
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + (
            [source.split("\n")[-1]] if source.split("\n")[-1] else []
        )
        if cell["source"] and not cell["source"][-1].endswith("\n") and source.endswith("\n"):
            cell["source"][-1] += "\n"
        break
    else:
        raise SystemExit("notebook 05 call was not found")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("patched notebook 05 call")


def write_provenance_notebook() -> None:
    path = ROOT / "notebooks" / "03b_data_provenance.ipynb"
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ۳ب. قیف data provenance قبل از مدل\n",
                "\n",
                "این نوت‌بوک را قبل از دفترچه‌های ۰۴ تا ۰۷ اجرا کنید. مدل فقط جعبهٔ analyzed را می‌بیند.\n",
                "\n",
                "شکل `data_provenance.png` در مخزن رسمی این جعبه‌ها را دارد: سمعی ۶۷۲ هم‌تراز و ۶۴۲ تحلیل‌شده، اسیلومتریک ۵۴۸ هم‌تراز و ۴۸۳ تحلیل‌شده.\n",
                "\n",
                "رضایت، انصراف، خرابی دستگاه، خرابی فایل، و شکست هم‌ترازی دو ECG در فایل منتشرشده نیستند. پوشه‌های موج از جعبهٔ هم‌تراز شروع می‌شوند. در موجودی قبلی، پوشهٔ موج سمعی ۶۷۲ و اسیلومتریک ۵۴۸ بود.\n",
                "\n",
                "جعبهٔ analyzed همان `in_feature_table` است. کسی که موج دارد و در این فهرست نیست، حذف کیفیت همان شکل است. برچسب «۳۱» روی شکل با تفاضل ۶۷۲ و ۶۴۲ یکی نیست؛ تفاضل ۳۰ است. این نوت‌بوک عدد مشاهده‌شده را چاپ می‌کند و برای رسیدن به برچسب، کسی را حذف نمی‌کند.\n",
                "\n",
                "حداقل ۱۷ اندازه‌گیری شکل ۷ بعد از این قیف است. این نوت‌بوک آن تعداد را ۲۲۷ نمی‌کند.\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "except ImportError:\n",
                "    print('خارج از کولب؛ اتصال درایو رد شد.')\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import base64\n",
                "from pathlib import Path\n",
                "root = Path('/content/aurorabp_reproduction')\n",
                "(root / 'src').mkdir(parents=True, exist_ok=True)\n",
                "(root / 'src' / '__init__.py').write_text('__version__ = \"0.1.0\"\\n', encoding='utf-8')\n",
                f"(root / 'src' / 'bp_protocol.py').write_bytes(base64.b64decode(\"{PAYLOAD}\"))\n",
                "print(root / 'src' / 'bp_protocol.py')\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import json\n",
                "import sys\n",
                "from pathlib import Path\n",
                "sys.path.insert(0, '/content/aurorabp_reproduction')\n",
                "import pandas as pd\n",
                "from src.bp_protocol import provenance_report, restrict_to_analyzed\n",
                "\n",
                "data_dir = Path('/content/drive/MyDrive/aurora_dataset')\n",
                "participants = pd.read_csv(data_dir / 'participants.tsv', sep='\\t')\n",
                "features = pd.read_csv(data_dir / 'features.tsv', sep='\\t')\n",
                "auscultatory = pd.read_csv(data_dir / 'measurements_auscultatory.tsv', sep='\\t')\n",
                "oscillometric = pd.read_csv(data_dir / 'measurements_oscillometric.tsv', sep='\\t')\n",
                "report = provenance_report(participants, features, auscultatory, oscillometric)\n",
                "kept = restrict_to_analyzed(features, participants)\n",
                "output = Path('/content/aurorabp_outputs')\n",
                "output.mkdir(parents=True, exist_ok=True)\n",
                "pd.DataFrame({'pid': report['analyzed_ids']}).to_csv(output / 'analyzed_pids.tsv', sep='\\t', index=False)\n",
                "summary = {key: value for key, value in report.items() if key != 'analyzed_ids'}\n",
                "(output / 'data_provenance.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')\n",
                "print(output / 'analyzed_pids.tsv')\n",
                "print('feature rows after the analyzed gate:', len(kept))\n",
            ],
        },
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", path.name)


def smoke() -> None:
    import pandas as pd
    import sys
    sys.path.insert(0, str(ROOT))
    from src.bp_protocol import provenance_report, restrict_to_analyzed

    participants = pd.DataFrame(
        {
            "pid": ["a1", "a2", "o1"],
            "in_feature_table": [1, 0, 1],
        }
    )
    features = pd.DataFrame(
        {
            "pid": ["a1", "a1", "o1"],
            "phase": ["return", "initial", "ambulatory"],
            "measurement": ["Temporal challenge start 1", "Calibration start 1", "measurement_1"],
            "date_time": ["21-01-02 10:00:00", "21-01-01 10:00:00", "21-01-01 12:00:00"],
            "sbp": [120, 118, 130],
            "dbp": [80, 78, 85],
            "baseline_sbp": [118, 118, 128],
            "baseline_dbp": [78, 78, 82],
            "delta_sbp": [2, 0, 2],
            "delta_dbp": [2, 0, 3],
        }
    )
    auscultatory = pd.DataFrame({"pid": ["a1", "a2"], "waveforms_generated": [1, 1]})
    oscillometric = pd.DataFrame({"pid": ["o1"], "waveforms_generated": [1]})
    report = provenance_report(participants, features, auscultatory, oscillometric)
    kept = restrict_to_analyzed(features, participants)
    assert list(kept["pid"]) == ["a1", "a1", "o1"]
    assert report["analyzed"]["auscultatory"] == 1
    assert report["aligned_featurized"]["auscultatory"] == 2
    print("smoke ok")


if __name__ == "__main__":
    write_provenance_notebook()
    for notebook in NOTEBOOKS:
        if notebook.name == "05_ambulatory_24h.ipynb":
            patch_notebook_05()
        replace_payload(notebook)
    smoke()
