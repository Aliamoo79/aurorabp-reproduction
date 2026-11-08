import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bp_protocol import FEATURE_GROUPS, STATIC_FEATURES, shared_ambulatory_cohort

columns = []
for group_columns in FEATURE_GROUPS.values():
    columns.extend(group_columns)


def blank():
    row = {column: 0.0 for column in columns}
    row.update({column: 0.0 for column in STATIC_FEATURES})
    return row


ambulatory = []
for index in range(6):
    for _ in range(20):
        row = blank()
        row.update(
            pid=f"o{index}",
            delta_sbp=10.0,
            delta_dbp=10.0,
            clock_hour=12,
            quality_optical=0.9,
            quality_pressure=0.9,
        )
        ambulatory.append(row)
row = blank()
row.update(
    pid="oBad",
    delta_sbp=10.0,
    delta_dbp=10.0,
    clock_hour=12,
    quality_optical=0.2,
    quality_pressure=0.9,
)
ambulatory.extend([row] * 20)
shared = shared_ambulatory_cohort(pd.DataFrame(ambulatory))
assert shared["pid"].nunique() == 6, shared["pid"].nunique()
assert "oBad" not in set(shared["pid"])
print("shared", shared["pid"].nunique(), "rows", len(shared))
print("smoke ok")
