"""Inventory an Aurora-BP ZIP or extracted tree without loading waveforms.

The official sample is only a schema example. Counts must come from the
archive that is actually opened. [GITHUB]
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

EXPECTED_TABLES = (
    "participants.tsv",
    "measurements_auscultatory.tsv",
    "measurements_oscillometric.tsv",
    "features.tsv",
)
WAVEFORM_COLUMNS = ("t", "ekg", "optical", "pressure", "accel_x", "accel_y", "accel_z")
EXERCISE_TOKENS = ("exercise", "running", "walking", "cool down", "cool_down")
QUALITY_THRESHOLD = 0.65  # [GITHUB]


def _normalize(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def _common_root(names: Iterable[str]) -> str:
    parts = [name.split("/") for name in names if name and not name.endswith("/")]
    if not parts:
        return ""
    prefix: List[str] = []
    for pieces in zip(*parts):
        if len(set(pieces)) != 1:
            break
        prefix.append(pieces[0])
    return "/".join(prefix)


class DatasetSource:
    """Read small TSV members from a ZIP or a directory. Waveforms stay on disk."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.kind = "zip" if self.path.is_file() else "dir"
        if self.kind == "zip":
            self._zip = zipfile.ZipFile(self.path)
            self.names = [_normalize(name) for name in self._zip.namelist() if not name.endswith("/")]
        elif self.path.is_dir():
            self._zip = None
            # One directory level, plus TSV files directly inside a child folder.
            # Do not walk extracted waveform trees; those belong in the ZIP index.
            names: List[str] = []
            for item in self.path.iterdir():
                if item.is_file():
                    names.append(item.name)
                elif item.is_dir():
                    for child in item.glob("*.tsv"):
                        names.append(f"{item.name}/{child.name}")
            self.names = names
        else:
            raise FileNotFoundError(path)
        self.root = _common_root(self.names)
        self._suffixes = set()
        for name in self.names:
            parts = name.split("/")
            for start in range(len(parts)):
                self._suffixes.add("/".join(parts[start:]))

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def _strip(self, name: str) -> str:
        name = _normalize(name)
        if self.root and (name == self.root or name.startswith(self.root + "/")):
            return name[len(self.root) + 1 :]
        return name

    def find(self, suffix: str) -> Optional[str]:
        matches = [name for name in self.names if self._strip(name) == suffix or name.endswith("/" + suffix)]
        if not matches:
            return None
        return sorted(matches, key=len)[0]

    def read_bytes(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def read_tsv(self, suffix: str) -> Optional[pd.DataFrame]:
        name = self.find(suffix)
        if name is None:
            return None
        return pd.read_csv(io.BytesIO(self.read_bytes(name)), sep="\t")

    def has(self, relative_name: str) -> bool:
        relative_name = _normalize(relative_name).lstrip("/")
        return relative_name in self._suffixes


def _quality_counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if column not in frame.columns:
        return {"present": 0}
    values = pd.to_numeric(frame[column], errors="coerce")
    return {
        "non_null": int(values.notna().sum()),
        "sentinel_minus_one": int((values == -1).sum()),
        "at_least_0.65": int((values >= QUALITY_THRESHOLD).sum()),
        "below_0.65_excluding_sentinel": int(((values >= 0) & (values < QUALITY_THRESHOLD)).sum()),
    }


def _exercise_mask(measurements: pd.Series) -> pd.Series:
    text = measurements.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=measurements.index)
    for token in EXERCISE_TOKENS:
        mask = mask | text.str.contains(token, regex=False)
    return mask


def inspect_dataset(path: str, waveform_sample: bool = True) -> Dict[str, object]:
    """Build a dataset inventory. This does not extract the archive."""
    source = DatasetSource(path)
    try:
        tables = {name: source.read_tsv(name) for name in EXPECTED_TABLES}
        participants = tables["participants.tsv"]
        auscultatory = tables["measurements_auscultatory.tsv"]
        oscillometric = tables["measurements_oscillometric.tsv"]
        features = tables["features.tsv"]

        report: Dict[str, object] = {
            "source": str(source.path),
            "source_kind": source.kind,
            "archive_root": source.root,
            "file_count": len(source.names),
            "missing_tables": [name for name, frame in tables.items() if frame is None],
            "top_level_entries": sorted({name.split("/")[0] for name in source.names}),
        }
        if participants is not None:
            report["n_participants"] = int(participants["pid"].nunique())
            report["n_auscultatory_participants"] = int(participants["pid"].astype(str).str.startswith("a").sum())
            report["n_oscillometric_participants"] = int(participants["pid"].astype(str).str.startswith("o").sum())
            if "in_feature_table" in participants.columns:
                flag = participants["in_feature_table"]
                report["n_in_feature_table"] = int(flag.fillna(0).astype(int).sum())
        if features is not None:
            report["n_feature_rows"] = int(len(features))
            report["feature_phases"] = features["phase"].value_counts(dropna=False).to_dict()
            report["n_feature_participants"] = int(features["pid"].nunique())

        measurement_reports = []
        for protocol, frame in (("auscultatory", auscultatory), ("oscillometric", oscillometric)):
            if frame is None:
                measurement_reports.append({"protocol": protocol, "missing": True})
                continue
            paths = frame.get("waveform_file_path", pd.Series(index=frame.index, dtype=object))
            present = paths.fillna("").map(lambda item: bool(item) and source.has(str(item)))
            exercise = _exercise_mask(frame["measurement"]) if "measurement" in frame.columns else pd.Series(False, index=frame.index)
            entry = {
                "protocol": protocol,
                "n_measurements": int(len(frame)),
                "phases": frame["phase"].value_counts(dropna=False).to_dict() if "phase" in frame.columns else {},
                "n_missing_waveform_path": int(paths.isna().sum() + (paths == "").sum()),
                "n_waveform_path_unresolved": int((paths.notna() & (paths != "") & ~present).sum()),
                "n_waveform_files_resolved": int(present.sum()),
                "n_sbp_present": int(pd.to_numeric(frame["sbp"], errors="coerce").notna().sum()) if "sbp" in frame.columns else 0,
                "n_dbp_present": int(pd.to_numeric(frame["dbp"], errors="coerce").notna().sum()) if "dbp" in frame.columns else 0,
                "pressure_quality": _quality_counts(frame, "pressure_quality"),
                "optical_quality": _quality_counts(frame, "optical_quality"),
                "n_exercise_named_rows": int(exercise.sum()),
                "duration_seconds": frame["duration"].describe().to_dict() if "duration" in frame.columns else {},
            }
            if "waveforms_generated" in frame.columns:
                generated = pd.to_numeric(frame["waveforms_generated"], errors="coerce")
                entry["n_waveforms_generated_zero"] = int((generated == 0).sum())
            measurement_reports.append(entry)
        report["measurements"] = measurement_reports

        if waveform_sample:
            waveform_names = [
                name
                for name in source.names
                if name.endswith(".tsv")
                and (
                    "/measurements_auscultatory/" in "/" + name
                    or "/measurements_oscillometric/" in "/" + name
                )
            ]
            if waveform_names:
                sample_name = waveform_names[0]
                waveform = pd.read_csv(io.BytesIO(source.read_bytes(sample_name)), sep="\t", nrows=20)
                time = pd.to_numeric(waveform["t"], errors="coerce") if "t" in waveform.columns else pd.Series(dtype=float)
                step = np.diff(time.to_numpy(dtype=float)) if len(time) > 1 else np.array([])
                report["waveform_sample"] = {
                    "file": source._strip(sample_name),
                    "columns": list(waveform.columns),
                    "missing_expected_columns": [column for column in WAVEFORM_COLUMNS if column not in waveform.columns],
                    "median_dt_seconds": float(np.median(step)) if step.size else None,
                    "estimated_hz": float(1.0 / np.median(step)) if step.size and np.median(step) > 0 else None,
                }
        return report
    finally:
        source.close()


def _waveform_name_summary(source: DatasetSource) -> Dict[str, object]:
    """Count waveform members from filenames. Does not read waveform contents."""
    by_phase: Dict[str, int] = {}
    pids = set()
    waveform_files = 0
    for name in source.names:
        if not (
            "/measurements_auscultatory/" in "/" + name
            or "/measurements_oscillometric/" in "/" + name
        ):
            continue
        if not name.endswith(".tsv"):
            continue
        waveform_files += 1
        stem = name.split("/")[-1][:-4]
        bits = stem.split(".", 2)
        if len(bits) >= 2:
            pids.add(bits[0])
            by_phase[bits[1]] = by_phase.get(bits[1], 0) + 1
    return {
        "source": str(source.path),
        "member_count": len(source.names),
        "archive_root": source.root,
        "n_waveform_files": waveform_files,
        "n_participant_folders": len(pids),
        "phases": by_phase,
        "example_members": source.names[:8],
    }


def inspect_collection(dataset_dir: str, zip_paths: List[str], waveform_sample: bool = True) -> Dict[str, object]:
    """Inventory the Drive folder and the two protocol ZIP archives.

    The released files for this copy are split by protocol. That split is not
    described as two archives in the paper. [DECISION] follow the paths given
    for this copy. Nothing is extracted.
    """
    sources: List[DatasetSource] = []
    errors: List[Dict[str, str]] = []
    directory = Path(dataset_dir)
    if dataset_dir and directory.is_dir():
        sources.append(DatasetSource(dataset_dir))
    else:
        errors.append({"path": dataset_dir, "error": "dataset directory is missing"})
    for zip_path in zip_paths:
        if not Path(zip_path).is_file():
            errors.append({"path": zip_path, "error": "zip file is missing"})
            continue
        try:
            sources.append(DatasetSource(zip_path))
        except zipfile.BadZipFile as exc:
            errors.append({"path": zip_path, "error": f"bad zip: {exc}"})
    try:
        tables = {}
        table_origins = {}
        for suffix in EXPECTED_TABLES:
            frame = None
            origin = None
            for source in sources:
                frame = source.read_tsv(suffix)
                if frame is not None:
                    origin = str(source.path)
                    break
            tables[suffix] = frame
            table_origins[suffix] = origin

        report: Dict[str, object] = {
            "dataset_dir": dataset_dir,
            "zip_paths": list(zip_paths),
            "open_errors": errors,
            "table_origins": table_origins,
            "missing_tables": [name for name, frame in tables.items() if frame is None],
            "archives": [_waveform_name_summary(source) for source in sources if source.kind == "zip"],
            "directory_files": sorted(sources[0].names) if sources and sources[0].kind == "dir" else [],
        }
        participants = tables["participants.tsv"]
        features = tables["features.tsv"]
        if participants is not None:
            report["n_participants"] = int(participants["pid"].nunique())
            report["n_auscultatory_participants"] = int(participants["pid"].astype(str).str.startswith("a").sum())
            report["n_oscillometric_participants"] = int(participants["pid"].astype(str).str.startswith("o").sum())
            if "in_feature_table" in participants.columns:
                report["n_in_feature_table"] = int(participants["in_feature_table"].fillna(0).astype(int).sum())
        if features is not None:
            report["n_feature_rows"] = int(len(features))
            report["n_feature_participants"] = int(features["pid"].nunique())
            report["feature_phases"] = features["phase"].value_counts(dropna=False).to_dict()

        measurement_reports = []
        for protocol, suffix in (
            ("auscultatory", "measurements_auscultatory.tsv"),
            ("oscillometric", "measurements_oscillometric.tsv"),
        ):
            frame = tables[suffix]
            if frame is None:
                measurement_reports.append({"protocol": protocol, "missing": True})
                continue
            paths = frame.get("waveform_file_path", pd.Series(index=frame.index, dtype=object))
            present = paths.fillna("").map(lambda item: bool(item) and any(source.has(str(item)) for source in sources))
            exercise = _exercise_mask(frame["measurement"]) if "measurement" in frame.columns else pd.Series(False, index=frame.index)
            entry: Dict[str, object] = {
                "protocol": protocol,
                "n_measurements": int(len(frame)),
                "phases": frame["phase"].value_counts(dropna=False).to_dict() if "phase" in frame.columns else {},
                "n_missing_waveform_path": int(paths.isna().sum() + (paths.fillna("") == "").sum()),
                "n_waveform_path_unresolved": int((paths.notna() & (paths.fillna("") != "") & ~present).sum()),
                "n_waveform_files_resolved": int(present.sum()),
                "n_sbp_present": int(pd.to_numeric(frame["sbp"], errors="coerce").notna().sum()) if "sbp" in frame.columns else 0,
                "n_dbp_present": int(pd.to_numeric(frame["dbp"], errors="coerce").notna().sum()) if "dbp" in frame.columns else 0,
                "pressure_quality": _quality_counts(frame, "pressure_quality"),
                "optical_quality": _quality_counts(frame, "optical_quality"),
                "n_exercise_named_rows": int(exercise.sum()),
            }
            if "waveforms_generated" in frame.columns:
                generated = pd.to_numeric(frame["waveforms_generated"], errors="coerce")
                entry["n_waveforms_generated_zero"] = int((generated == 0).sum())
            measurement_reports.append(entry)
        report["measurements"] = measurement_reports

        if waveform_sample:
            samples = []
            for source in sources:
                if source.kind != "zip":
                    continue
                waveform_names = [
                    name
                    for name in source.names
                    if name.endswith(".tsv")
                    and (
                        "/measurements_auscultatory/" in "/" + name
                        or "/measurements_oscillometric/" in "/" + name
                    )
                ]
                if not waveform_names:
                    continue
                sample_name = waveform_names[0]
                waveform = pd.read_csv(io.BytesIO(source.read_bytes(sample_name)), sep="\t", nrows=20)
                time = pd.to_numeric(waveform["t"], errors="coerce") if "t" in waveform.columns else pd.Series(dtype=float)
                step = np.diff(time.to_numpy(dtype=float)) if len(time) > 1 else np.array([])
                samples.append(
                    {
                        "archive": str(source.path),
                        "file": source._strip(sample_name),
                        "columns": list(waveform.columns),
                        "missing_expected_columns": [column for column in WAVEFORM_COLUMNS if column not in waveform.columns],
                        "median_dt_seconds": float(np.median(step)) if step.size else None,
                        "estimated_hz": float(1.0 / np.median(step)) if step.size and np.median(step) > 0 else None,
                    }
                )
            report["waveform_samples"] = samples
        return report
    finally:
        for source in sources:
            source.close()


def write_inventory(report: Dict[str, object], output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
