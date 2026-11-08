import base64
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
for path in sorted(root.glob("*.ipynb")):
    nb = json.loads(path.read_text(encoding="utf-8"))
    blob = "".join("".join(cell["source"]) for cell in nb["cells"])
    match = re.search(r'b64decode\("([^"]+)"\)', blob)
    if not match:
        print(path.name, "NO PAYLOAD")
        continue
    code = base64.b64decode(match.group(1)).decode("utf-8")
    print(
        path.name,
        "hypothesis", "bias hypothesis" in code,
        "pool", "post_calibration_pool" in code,
        "shared", "shared_ambulatory_cohort" in code,
        "return_only", 'phase"].astype(str) == "return"' in code or "phase\"].astype(str) == \"return\"" in code,
    )
