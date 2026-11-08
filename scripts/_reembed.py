import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refresh_notebook_protocol as refresh

for path in refresh.NOTEBOOKS:
    refresh.replace_payload(path)
