"""Return-visit blood pressure protocol for the Aurora-BP reproduction.

The sensor features are the released delta columns, not features computed
by this project. Ridge alpha, error degrees of freedom, and the ECG quality
rule are not author settings.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# [DECISION] Standardized inputs, unpenalized intercept, alpha 1.
# The paper does not give alpha. The example notebook uses 0.001 on raw units.
RIDGE_ALPHA = 1.0
QUALITY_THRESHOLD = 0.65  # [GITHUB] docs: used in the study's prediction experiments
BOOTSTRAP_REPLICATES = 100  # [PAPER]
BOOTSTRAP_FRACTION = 0.90  # [PAPER]
# [DECISION] The paper does not say sampling with replacement.
BOOTSTRAP_WITH_REPLACEMENT = False
YOUNG_AGE_MAX = 35  # [PAPER]
# [PAPER] methods. Figure 3 caption uses a strict inequality.
NORMOTENSIVE_RESTING_SBP_MAX = 130.0
EXERCISE_PATTERN = re.compile(r"exercise|running|walking|cool.?down", re.IGNORECASE)

# Table IV, using the released delta columns. [PAPER] group membership.
# [GITHUB] the example model predicts delta_sbp from delta features.
# Time-of-day terms are absolute, not deltas. [PAPER]
FEATURE_GROUPS: Dict[str, List[str]] = {
    "ecg": [
        "delta_beat_length_ekg",
        "delta_hr_ekg",
        "delta_hrv_ekg",
        "cos_hour",
        "sin_hour",
    ],
    "optical": [
        "delta_arm_angle",
        "delta_beat_length_ekg",
        "delta_dpdt_optical",
        "delta_hr_ekg",
        "delta_hr_optical",
        "delta_hrv_ekg",
        "delta_quality_optical",
        "delta_invpat_optical",
        "delta_rpat_optical",
        "cos_hour",
        "sin_hour",
    ],
    "smartwatch": [
        "delta_arm_angle",
        "delta_dpdt_optical",
        "delta_hr_optical",
        "delta_quality_optical",
        "cos_hour",
        "sin_hour",
    ],
    "tonometric": [
        "delta_arm_angle",
        "delta_aix_pressure",
        "delta_beat_length_ekg",
        "delta_dicrotic_notch_height_pressure",
        "delta_dpdt_pressure",
        "delta_ejection_duration_pressure",
        "delta_ejection_duration_fraction_pressure",
        "delta_hr_ekg",
        "delta_hr_pressure",
        "delta_hrv_ekg",
        "delta_quality_pressure",
        "delta_invpat_pressure",
        "delta_rpat_pressure",
        "delta_rwat_pressure",
        "delta_sysrise_pressure",
        "cos_hour",
        "sin_hour",
    ],
}

# [GITHUB] The released table has no optical or tonometric HRV column.
# Table IV asks for optical HRV in the optical and smartwatch groups.
MISSING_TABLE_FEATURES = {
    "optical": ["hrv_optical"],
    "smartwatch": ["hrv_optical"],
}

# [DECISION] 0.65 is documented on the optical and tonometric quality columns.
# The ECG group has no quality term, and ECG availability in the paper is higher
# than the other sensors, so that threshold is not applied to the ECG group.
QUALITY_COLUMN = {
    "ecg": None,
    "optical": "quality_optical",
    "smartwatch": "quality_optical",
    "tonometric": "quality_pressure",
}

STATIC_FEATURES = ["baseline_sbp", "baseline_dbp", "cos_hour", "sin_hour"]

# Boxes in docs/data_provenance.png. [GITHUB]
# Consent, opt-out, miswear, file failure, and the two-ECG alignment discard
# are not rows in the released tables. Waveform folders begin at the aligned box.
PROVENANCE_BOXES = {
    "auscultatory": {"aligned_featurized": 672, "analyzed": 642},
    "oscillometric": {"aligned_featurized": 548, "analyzed": 483},
}


def _protocol_counts(identifiers) -> Dict[str, int]:
    values = pd.Index(pd.unique(pd.Series(identifiers).astype(str)))
    return {
        "auscultatory": int(values.str.startswith("a").sum()),
        "oscillometric": int(values.str.startswith("o").sum()),
        "total": int(values.size),
    }


def aligned_participant_ids(*measurement_frames: Optional[pd.DataFrame]) -> pd.Index:
    """Participants with at least one generated waveform. [GITHUB] aligned, featurized."""
    kept: List[pd.Series] = []
    for frame in measurement_frames:
        if frame is None or frame.empty or "pid" not in frame.columns:
            continue
        if "waveforms_generated" in frame.columns:
            generated = pd.to_numeric(frame["waveforms_generated"], errors="coerce")
            kept.append(frame.loc[generated.eq(1), "pid"].astype(str))
        elif "waveform_file_path" in frame.columns:
            path = frame["waveform_file_path"]
            present = path.notna() & path.astype(str).str.strip().ne("") & path.astype(str).ne("nan")
            kept.append(frame.loc[present, "pid"].astype(str))
    if not kept:
        return pd.Index([])
    return pd.Index(pd.concat(kept, ignore_index=True).unique())


def analyzed_participant_ids(participants: pd.DataFrame, features: pd.DataFrame) -> pd.Index:
    """Final provenance box. [GITHUB] in_feature_table is the released analyzed cohort."""
    if participants is not None and "in_feature_table" in participants.columns:
        flag = participants["in_feature_table"]
        if flag.dtype == bool:
            keep = flag.fillna(False)
        else:
            numeric = pd.to_numeric(flag, errors="coerce")
            if numeric.notna().any():
                keep = numeric.fillna(0).eq(1)
            else:
                keep = flag.astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        return pd.Index(participants.loc[keep, "pid"].astype(str).unique())
    return pd.Index(features["pid"].astype(str).unique())


def provenance_report(
    participants: pd.DataFrame,
    features: pd.DataFrame,
    auscultatory_measurements: Optional[pd.DataFrame] = None,
    oscillometric_measurements: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Compare the released files with the provenance figure.

    Stages above the aligned box cannot be replayed: those participants are
    not in the release. Quality failure is the aligned participant who is
    absent from in_feature_table.
    """
    released = _protocol_counts(participants["pid"])
    analyzed = analyzed_participant_ids(participants, features)
    analyzed_counts = _protocol_counts(analyzed)
    feature_counts = _protocol_counts(features["pid"])
    aligned = aligned_participant_ids(auscultatory_measurements, oscillometric_measurements)
    aligned_counts = _protocol_counts(aligned) if aligned.size else None
    report: Dict[str, object] = {
        "released_participants": released,
        "feature_table_participants": feature_counts,
        "analyzed": analyzed_counts,
        "aligned_featurized": aligned_counts,
        "paper_boxes": PROVENANCE_BOXES,
        "analyzed_ids": analyzed,
    }
    print("provenance: released participants", released, flush=True)
    if aligned_counts is not None:
        print("provenance: aligned, featurized (waveforms_generated==1)", aligned_counts, flush=True)
        for protocol, boxes in PROVENANCE_BOXES.items():
            observed = aligned_counts[protocol]
            expected = boxes["aligned_featurized"]
            print(f"provenance: {protocol} aligned {observed} versus figure {expected}", flush=True)
    print("provenance: analyzed (in_feature_table)", analyzed_counts, flush=True)
    print("provenance: distinct pids inside features.tsv", feature_counts, flush=True)
    for protocol, boxes in PROVENANCE_BOXES.items():
        observed = analyzed_counts[protocol]
        expected = boxes["analyzed"]
        print(f"provenance: {protocol} analyzed {observed} versus figure {expected}", flush=True)
        if aligned_counts is not None:
            failed = aligned_counts[protocol] - observed
            print(f"provenance: {protocol} quality-failure participants {failed}", flush=True)
    outside = int((~participants["pid"].astype(str).isin(set(analyzed))).sum())
    print(f"provenance: participants outside the analyzed box {outside}", flush=True)
    return report


def restrict_to_analyzed(features: pd.DataFrame, participants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Keep feature rows for the analyzed provenance box before any model fit."""
    if participants is None:
        print("provenance: participants.tsv was not passed; features.tsv pids are used as-is", flush=True)
        return features
    report = provenance_report(participants, features)
    analyzed = set(report["analyzed_ids"])
    kept = features.loc[features["pid"].astype(str).isin(analyzed)].copy()
    print(
        f"provenance: model input {kept['pid'].nunique()} participants, {len(kept)} feature rows",
        flush=True,
    )
    return kept


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Clock hour from the timestamp, without timezone conversion. [DECISION]"""
    output = frame.copy()
    stamps = pd.to_datetime(output["date_time"], errors="coerce")
    hour = stamps.dt.hour + stamps.dt.minute / 60.0 + stamps.dt.second / 3600.0
    angle = 2.0 * np.pi * hour / 24.0
    output["cos_hour"] = np.cos(angle)
    output["sin_hour"] = np.sin(angle)
    return output


def exercise_mask(measurements: pd.Series) -> pd.Series:
    return measurements.astype(str).str.contains(EXERCISE_PATTERN, na=False)


def _ridge_apply(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    """Standardize on the training rows and keep the intercept unpenalized."""
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_z = (train_x - center) / scale
    target_center = float(train_y.mean())
    penalty = alpha * np.eye(train_x.shape[1])
    coefficient = np.linalg.solve(
        train_z.T @ train_z + penalty,
        train_z.T @ (train_y - target_center),
    )
    return ((test_x - center) / scale) @ coefficient + target_center


def lopo_predict(values: np.ndarray, target: np.ndarray, groups: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    """Leave-one-participant-out ridge. Train-fold standardization. [DECISION] alpha."""
    features = np.asarray(values, dtype=float)
    observed = np.asarray(target, dtype=float)
    labels = np.asarray(groups)
    predicted = np.full(observed.shape, np.nan)
    for label in pd.unique(labels):
        test = labels == label
        train = ~test
        if int(train.sum()) < features.shape[1] + 1:
            continue
        predicted[test] = _ridge_apply(features[train], observed[train], features[test], alpha)
    return predicted


def _error_summary(predicted: np.ndarray, observed: np.ndarray) -> Dict[str, float]:
    error = predicted - observed
    finite = np.isfinite(error)
    error = error[finite]
    if error.size < 2:
        return {"n": int(error.size), "bias": np.nan, "sd": np.nan}
    return {
        "n": int(error.size),
        "bias": float(error.mean()),
        "sd": float(error.std(ddof=1)),
    }


def _bootstrap_ids(identifiers: np.ndarray, replicates: int, fraction: float, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    count = int(round(identifiers.size * fraction))
    count = min(max(count, 2), identifiers.size)
    for _ in range(replicates):
        yield rng.choice(identifiers, size=count, replace=BOOTSTRAP_WITH_REPLACEMENT)


def complete_cases(frame: pd.DataFrame, columns: List[str], target: str) -> pd.DataFrame:
    """Rows used together for the feature model and the paired baselines."""
    needed = list(dict.fromkeys(columns + STATIC_FEATURES + [target, "pid"]))
    return frame.dropna(subset=needed).reset_index(drop=True)


def evaluate_rows(
    frame: pd.DataFrame,
    columns: List[str],
    target: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 2022,
    fit: bool = True,
) -> Dict[str, float]:
    """Measurement-level error. Headline bias and SD are means over bootstraps. [PAPER]"""
    subset = frame
    observed = subset[target].to_numpy(dtype=float)
    labels = subset["pid"].to_numpy()
    matrix = subset[columns].to_numpy(dtype=float) if fit else None
    if fit:
        predicted = lopo_predict(matrix, observed, labels)
    else:
        predicted = np.zeros(observed.shape, dtype=float)
    full = _error_summary(predicted, observed)
    biases: List[float] = []
    deviations: List[float] = []
    identifiers = np.unique(labels)
    for chosen in _bootstrap_ids(identifiers, replicates, BOOTSTRAP_FRACTION, seed):
        chosen_rows = np.isin(labels, chosen)
        if int(chosen_rows.sum()) < 2:
            continue
        if fit:
            fold_prediction = lopo_predict(matrix[chosen_rows], observed[chosen_rows], labels[chosen_rows])
        else:
            fold_prediction = np.zeros(int(chosen_rows.sum()), dtype=float)
        summary = _error_summary(fold_prediction, observed[chosen_rows])
        if np.isfinite(summary["bias"]) and np.isfinite(summary["sd"]):
            biases.append(summary["bias"])
            deviations.append(summary["sd"])
    bias_array = np.asarray(biases, dtype=float)
    sd_array = np.asarray(deviations, dtype=float)
    return {
        "n_rows": int(len(subset)),
        "n_participants": int(identifiers.size),
        "lopo_bias": full["bias"],
        "lopo_sd": full["sd"],
        "boot_bias": float(bias_array.mean()) if bias_array.size else np.nan,
        "boot_sd": float(sd_array.mean()) if sd_array.size else np.nan,
        "boot_bias_p2_5": float(np.percentile(bias_array, 2.5)) if bias_array.size else np.nan,
        "boot_bias_p97_5": float(np.percentile(bias_array, 97.5)) if bias_array.size else np.nan,
        "boot_replicates_used": int(bias_array.size),
    }


def resting_supine_sbp(features: pd.DataFrame) -> Optional[pd.Series]:
    """Median initial supine SBP. [DECISION] The paper says resting SBP and does not name this rule."""
    initial = features.loc[features["phase"].astype(str) == "initial"]
    supine = initial.loc[initial["measurement"].astype(str).str.contains("supine", case=False, na=False)]
    if supine.empty or "sbp" not in supine.columns:
        return None
    return supine.groupby("pid")["sbp"].median()


def prepare_return_visit(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = add_time_features(features)
    return_rows = frame.loc[frame["phase"].astype(str) == "return"].copy()
    dropped = exercise_mask(return_rows["measurement"])
    kept = return_rows.loc[~dropped].copy()
    return kept, return_rows.loc[dropped, "measurement"].astype(str)


def population_frame(frame: pd.DataFrame, participants: pd.DataFrame, population: str, resting: Optional[pd.Series]) -> pd.DataFrame:
    if population == "all":
        return frame
    if population == "young":
        ages = participants.loc[participants["age"].notna(), ["pid", "age"]]
        young = ages.loc[ages["age"] <= YOUNG_AGE_MAX, "pid"]
        return frame.loc[frame["pid"].isin(young)]
    if population == "normotensive":
        if resting is None:
            return frame.iloc[0:0]
        kept = resting.loc[resting <= NORMOTENSIVE_RESTING_SBP_MAX].index
        return frame.loc[frame["pid"].isin(kept)]
    raise ValueError(population)


def quality_filter(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    column = QUALITY_COLUMN[group]
    if column is None:
        return frame
    values = frame[column]
    return frame.loc[values >= QUALITY_THRESHOLD]


def run_return_visit(
    features: pd.DataFrame,
    participants: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 2022,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fit separate return-visit models by protocol, population, and feature group."""
    features = restrict_to_analyzed(features, participants)
    visit, dropped_names = prepare_return_visit(features)
    resting = resting_supine_sbp(features)
    print(
        "return visit in-lab only: phase=return, "
        f"{visit['pid'].nunique()} participants, {len(visit)} rows; ambulatory excluded",
        flush=True,
    )
    rows: List[dict] = []
    protocols = {
        "auscultatory": visit.loc[visit["pid"].astype(str).str.startswith("a")],
        "oscillometric": visit.loc[visit["pid"].astype(str).str.startswith("o")],
    }
    for protocol, protocol_rows in protocols.items():
        for population in ("all", "young", "normotensive"):
            people = population_frame(protocol_rows, participants, population, resting)
            if population == "normotensive" and resting is None:
                rows.append(
                    {
                        "protocol": protocol,
                        "population": population,
                        "row_set": "unavailable",
                        "model": "unavailable",
                        "target": "",
                        "reliability": "resting supine SBP could not be identified",
                        "n_rows": 0,
                    }
                )
                continue
            for group, columns in FEATURE_GROUPS.items():
                available = quality_filter(people, group)
                missing = MISSING_TABLE_FEATURES.get(group, [])
                if available.empty:
                    print(f"{protocol} {population} {group}: no rows", flush=True)
                    continue
                reliability = "official features; alpha and ECG quality rule are decisions"
                if missing:
                    reliability = "missing " + ", ".join(missing) + "; " + reliability
                if population == "normotensive":
                    reliability += "; resting SBP rule is a decision"
                for target in ("delta_sbp", "delta_dbp"):
                    usable = complete_cases(available, columns, target)
                    if usable["pid"].nunique() < 10:
                        print(f"{protocol} {population} {group} {target}: fewer than 10 participants", flush=True)
                        continue
                    ridge = evaluate_rows(usable, columns, target, replicates=replicates, seed=seed, fit=True)
                    zero = evaluate_rows(usable, columns, target, replicates=replicates, seed=seed, fit=False)
                    static = evaluate_rows(usable, STATIC_FEATURES, target, replicates=replicates, seed=seed, fit=True)
                    for model, summary in (("ridge", ridge), ("zero", zero), ("static", static)):
                        rows.append(
                            {
                                "protocol": protocol,
                                "population": population,
                                "row_set": group,
                                "model": model,
                                "target": target,
                                "missing_table_features": ",".join(missing),
                                "quality_column": QUALITY_COLUMN[group] or "",
                                "exercise_rows_removed": int(dropped_names.size),
                                "reliability": reliability,
                                **summary,
                            }
                        )
                    print(
                        f"{protocol} {population} {group} {target}: "
                        f"ridge {ridge['boot_bias']:.2f}±{ridge['boot_sd']:.2f}  "
                        f"zero {zero['boot_bias']:.2f}±{zero['boot_sd']:.2f}  "
                        f"static {static['boot_bias']:.2f}±{static['boot_sd']:.2f}  "
                        f"n={ridge['n_rows']}",
                        flush=True,
                    )
                    if output_path:
                        pd.DataFrame(rows).to_json(output_path, orient="records", indent=2)
    return pd.DataFrame(rows)


# [PAPER] Figure 7 keeps participants with at least 17 usable ambulatory measurements.
AMBULATORY_MINIMUM_MEASUREMENTS = 17


def prepare_ambulatory(features: pd.DataFrame) -> pd.DataFrame:
    """Oscillometric ambulatory rows only. Auscultatory has no ambulatory phase. [PAPER]"""
    frame = add_time_features(features)
    ambulatory = frame.loc[frame["phase"].astype(str) == "ambulatory"].copy()
    ambulatory = ambulatory.loc[ambulatory["pid"].astype(str).str.startswith("o")]
    ambulatory = ambulatory.loc[~exercise_mask(ambulatory["measurement"])].copy()
    stamps = pd.to_datetime(ambulatory["date_time"], errors="coerce")
    # [DECISION] The paper bins by hour of day. The bin is the clock hour, not the fractional hour.
    ambulatory["clock_hour"] = stamps.dt.hour
    return ambulatory


def cohort_with_minimum(frame: pd.DataFrame, columns: List[str], target: str) -> pd.DataFrame:
    usable = complete_cases(frame, columns, target)
    usable = usable.dropna(subset=["clock_hour"]).reset_index(drop=True)
    counts = usable.groupby("pid").size()
    kept = counts[counts >= AMBULATORY_MINIMUM_MEASUREMENTS].index
    return usable.loc[usable["pid"].isin(kept)].reset_index(drop=True)


def shared_ambulatory_cohort(ambulatory: pd.DataFrame) -> pd.DataFrame:
    """One Figure 7 cohort for every feature group and both baselines.

    A row must pass quality 0.65 on both optical and tonometry and have every
    released Table IV column. A participant needs at least 17 such rows.
    [DECISION] The paper states 227 people and does not spell this intersection.
    """
    required = ["quality_optical", "quality_pressure", "delta_sbp", "delta_dbp", "clock_hour"]
    present = ambulatory.dropna(subset=required)
    both = present.loc[
        (present["quality_optical"] >= QUALITY_THRESHOLD) & (present["quality_pressure"] >= QUALITY_THRESHOLD)
    ].copy()
    columns: List[str] = []
    for group_columns in FEATURE_GROUPS.values():
        for column in group_columns:
            if column in both.columns and column not in columns:
                columns.append(column)
    usable = complete_cases(both, columns, "delta_sbp")
    usable = usable.dropna(subset=["delta_dbp", "clock_hour"]).reset_index(drop=True)
    counts = usable.groupby("pid").size()
    kept = counts[counts >= AMBULATORY_MINIMUM_MEASUREMENTS].index
    return usable.loc[usable["pid"].isin(kept)].reset_index(drop=True)


def person_errors(frame: pd.DataFrame, predicted: np.ndarray, target: str) -> np.ndarray:
    """Hour-mean, then the mean of the available hours, then prediction minus reference. [PAPER]/[DECISION]

    Hours with no measurement are left out. The paper's phrase "mean of these 24 values"
    does not say whether an empty hour is imputed.
    """
    temporary = pd.DataFrame(
        {
            "pid": frame["pid"].to_numpy(),
            "hour": frame["clock_hour"].to_numpy(),
            "predicted": np.asarray(predicted, dtype=float),
            "observed": frame[target].to_numpy(dtype=float),
        }
    )
    hourly = temporary.groupby(["pid", "hour"], as_index=False)[["predicted", "observed"]].mean()
    person = hourly.groupby("pid", as_index=False)[["predicted", "observed"]].mean()
    return person["predicted"].to_numpy(dtype=float) - person["observed"].to_numpy(dtype=float)


def evaluate_24h(
    frame: pd.DataFrame,
    columns: List[str],
    target: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 2022,
    fit: bool = True,
) -> Dict[str, float]:
    """Participant-level 24-hour error. Headline numbers are means over bootstraps. [PAPER]"""
    observed_rows = frame[target].to_numpy(dtype=float)
    labels = frame["pid"].to_numpy()
    matrix = frame[columns].to_numpy(dtype=float) if fit else None
    if fit:
        predicted = lopo_predict(matrix, observed_rows, labels)
    else:
        predicted = np.zeros(observed_rows.shape, dtype=float)
    errors = person_errors(frame, predicted, target)
    full = {
        "bias": float(errors.mean()) if errors.size else np.nan,
        "sd": float(errors.std(ddof=1)) if errors.size > 1 else np.nan,
    }
    biases: List[float] = []
    deviations: List[float] = []
    identifiers = np.unique(labels)
    for chosen in _bootstrap_ids(identifiers, replicates, BOOTSTRAP_FRACTION, seed):
        chosen_rows = np.isin(labels, chosen)
        if int(np.unique(labels[chosen_rows]).size) < 10:
            continue
        if fit:
            fold_prediction = lopo_predict(matrix[chosen_rows], observed_rows[chosen_rows], labels[chosen_rows])
        else:
            fold_prediction = np.zeros(int(chosen_rows.sum()), dtype=float)
        fold_errors = person_errors(frame.iloc[np.flatnonzero(chosen_rows)].reset_index(drop=True), fold_prediction, target)
        if fold_errors.size < 2:
            continue
        biases.append(float(fold_errors.mean()))
        deviations.append(float(fold_errors.std(ddof=1)))
    bias_array = np.asarray(biases, dtype=float)
    sd_array = np.asarray(deviations, dtype=float)
    return {
        "n_rows": int(len(frame)),
        "n_participants": int(identifiers.size),
        "lopo_bias": full["bias"],
        "lopo_sd": full["sd"],
        "boot_bias": float(bias_array.mean()) if bias_array.size else np.nan,
        "boot_sd": float(sd_array.mean()) if sd_array.size else np.nan,
        "boot_bias_p2_5": float(np.percentile(bias_array, 2.5)) if bias_array.size else np.nan,
        "boot_bias_p97_5": float(np.percentile(bias_array, 97.5)) if bias_array.size else np.nan,
        "boot_replicates_used": int(bias_array.size),
    }


def run_ambulatory_24h(
    features: pd.DataFrame,
    participants: Optional[pd.DataFrame] = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 2022,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """24-hour averages for the full oscillometric ambulatory cohort. One population only."""
    features = restrict_to_analyzed(features, participants)
    ambulatory = prepare_ambulatory(features)
    shared = shared_ambulatory_cohort(ambulatory)
    print(
        f"shared 24h cohort: {shared['pid'].nunique()} participants, {len(shared)} rows; "
        f"trained inside these rows only (paper Figure 7 reports 227)",
        flush=True,
    )
    rows: List[dict] = []
    reliability = (
        "shared rows for every group: both sensors >=0.65, all released Table IV columns, "
        ">=17; model trained inside this cohort; empty hours omitted"
    )
    for group, columns in FEATURE_GROUPS.items():
        missing = MISSING_TABLE_FEATURES.get(group, [])
        group_reliability = reliability
        if missing:
            group_reliability = "missing " + ", ".join(missing) + "; " + reliability
        if shared["pid"].nunique() < 10:
            print(f"{group}: shared cohort has fewer than 10 participants", flush=True)
            continue
        for target in ("delta_sbp", "delta_dbp"):
            ridge = evaluate_24h(shared, columns, target, replicates=replicates, seed=seed, fit=True)
            zero = evaluate_24h(shared, columns, target, replicates=replicates, seed=seed, fit=False)
            static = evaluate_24h(shared, STATIC_FEATURES, target, replicates=replicates, seed=seed, fit=True)
            for model, summary in (("ridge", ridge), ("zero", zero), ("static", static)):
                rows.append(
                    {
                        "population": "all",
                        "row_set": group,
                        "model": model,
                        "target": target,
                        "minimum_measurements": AMBULATORY_MINIMUM_MEASUREMENTS,
                        "missing_table_features": ",".join(missing),
                        "reliability": group_reliability,
                        **summary,
                    }
                )
            print(
                f"{group} {target}: "
                f"ridge {ridge['boot_bias']:.2f}±{ridge['boot_sd']:.2f}  "
                f"zero {zero['boot_bias']:.2f}±{zero['boot_sd']:.2f}  "
                f"static {static['boot_bias']:.2f}±{static['boot_sd']:.2f}  "
                f"people={ridge['n_participants']} rows={ridge['n_rows']}",
                flush=True,
            )
            if output_path:
                pd.DataFrame(rows).to_json(output_path, orient="records", indent=2)
    return pd.DataFrame(rows)


def resting_reference_sbp(features: pd.DataFrame) -> pd.Series:
    """Median initial resting SBP per participant.

    Oscillometric rows are named Supine. Auscultatory supine calibration is
    named Calibration start. A participant with both uses the Supine rows.
    [DECISION] The paper says resting SBP and does not name these rows.
    """
    initial = features.loc[features["phase"].astype(str) == "initial"].copy()
    names = initial["measurement"].astype(str)
    supine = names.str.contains("supine", case=False, na=False)
    calibration = names.str.contains("calibration start", case=False, na=False)
    chosen = initial.loc[supine | calibration].copy()
    has_supine = set(initial.loc[supine, "pid"].astype(str))
    drop_calibration = chosen["pid"].astype(str).isin(has_supine) & chosen["measurement"].astype(str).str.contains(
        "calibration start", case=False, na=False
    )
    chosen = chosen.loc[~drop_calibration]
    if chosen.empty:
        return pd.Series(dtype=float)
    return chosen.groupby(chosen["pid"].astype(str))["sbp"].median()


def subset_ids(features: pd.DataFrame, participants: pd.DataFrame, population: str) -> pd.Index:
    """Participant ids for one Figure 7 subpopulation."""
    if population == "young":
        ages = participants.dropna(subset=["age"])
        return pd.Index(ages.loc[ages["age"] <= YOUNG_AGE_MAX, "pid"].astype(str).unique())
    if population == "normotensive":
        resting = resting_reference_sbp(features)
        kept = resting.loc[resting <= NORMOTENSIVE_RESTING_SBP_MAX]
        return pd.Index(kept.index.astype(str))
    raise ValueError(population)


def run_ambulatory_subsets(
    features: pd.DataFrame,
    participants: pd.DataFrame,
    populations: Iterable[str] = ("young", "normotensive"),
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 2022,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """24-hour models trained separately on each subpopulation. [PAPER]"""
    features = restrict_to_analyzed(features, participants)
    ambulatory = prepare_ambulatory(features)
    rows: List[dict] = []
    for population in populations:
        identifiers = set(subset_ids(features, participants, population).astype(str))
        people = ambulatory.loc[ambulatory["pid"].astype(str).isin(identifiers)].copy()
        shared = shared_ambulatory_cohort(people)
        print(
            f"{population}: shared cohort {shared['pid'].nunique()} participants, {len(shared)} rows; trained inside these rows only",
            flush=True,
        )
        reliability = (
            "separate subpopulation model; shared rows for every group; "
            "model trained inside this cohort; empty hours omitted"
        )
        if population == "normotensive":
            reliability += "; resting SBP <= 130 from Supine or Calibration start"
        for group, columns in FEATURE_GROUPS.items():
            missing = MISSING_TABLE_FEATURES.get(group, [])
            group_reliability = reliability
            if missing:
                group_reliability = "missing " + ", ".join(missing) + "; " + reliability
            if shared["pid"].nunique() < 10:
                print(f"{population} {group}: shared cohort has fewer than 10 participants", flush=True)
                continue
            for target in ("delta_sbp", "delta_dbp"):
                ridge = evaluate_24h(shared, columns, target, replicates=replicates, seed=seed, fit=True)
                zero = evaluate_24h(shared, columns, target, replicates=replicates, seed=seed, fit=False)
                static = evaluate_24h(shared, STATIC_FEATURES, target, replicates=replicates, seed=seed, fit=True)
                for model, summary in (("ridge", ridge), ("zero", zero), ("static", static)):
                    rows.append(
                        {
                            "population": population,
                            "row_set": group,
                            "model": model,
                            "target": target,
                            "minimum_measurements": AMBULATORY_MINIMUM_MEASUREMENTS,
                            "missing_table_features": ",".join(missing),
                            "reliability": group_reliability,
                            **summary,
                        }
                    )
                print(
                    f"{population} {group} {target}: "
                    f"ridge {ridge['boot_bias']:.2f}±{ridge['boot_sd']:.2f}  "
                    f"zero {zero['boot_bias']:.2f}±{zero['boot_sd']:.2f}  "
                    f"static {static['boot_bias']:.2f}±{static['boot_sd']:.2f}  "
                    f"people={ridge['n_participants']} rows={ridge['n_rows']}",
                    flush=True,
                )
                if output_path:
                    pd.DataFrame(rows).to_json(output_path, orient="records", indent=2)
    return pd.DataFrame(rows)


def assign_posture(measurements: pd.Series) -> pd.Series:
    """Map a measurement name to supine, seated, or other.

    [DECISION] Auscultatory names do not say supine. A name containing seat
    or sitting is seated. A remaining "challenge start" or "calibration start"
    name is treated as the supine series. Anything else stays other.
    """
    text = measurements.astype(str).str.lower()
    posture = pd.Series("other", index=measurements.index)
    posture.loc[text.str.contains("seat|sitting", regex=True, na=False)] = "seated"
    posture.loc[text.str.contains("supine", case=False, na=False)] = "supine"
    start = text.str.contains("challenge start|calibration start", regex=True, na=False)
    posture.loc[start & posture.eq("other")] = "supine"
    return posture


def _slice_summary(errors: np.ndarray) -> Dict[str, float]:
    if errors.size < 2:
        return {"n": int(errors.size), "bias": np.nan, "sd": np.nan}
    return {"n": int(errors.size), "bias": float(errors.mean()), "sd": float(errors.std(ddof=1))}


def run_return_posture(
    features: pd.DataFrame,
    participants: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 2022,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Auscultatory return-visit SBP, then split the same errors by posture.

    Table VII reports standard deviations. The All column matches the pooled
    return-visit result, so posture columns are slices, not new models. [PAPER]
    """
    features = restrict_to_analyzed(features, participants)
    visit, _dropped = prepare_return_visit(features)
    visit = visit.loc[visit["pid"].astype(str).str.startswith("a")].copy()
    visit["posture"] = assign_posture(visit["measurement"])
    print(visit.groupby(["posture", "measurement"]).size().sort_values(ascending=False).head(30).to_string(), flush=True)
    rows: List[dict] = []
    for population in ("all", "normotensive", "young"):
        if population == "all":
            people = visit
        else:
            people = visit.loc[visit["pid"].astype(str).isin(subset_ids(features, participants, population))].copy()
        for group, columns in FEATURE_GROUPS.items():
            available = quality_filter(people, group)
            usable = complete_cases(available, columns, "delta_sbp")
            if "posture" not in usable.columns:
                print(f"{population} {group}: posture column dropped", flush=True)
                continue
            if usable["pid"].nunique() < 10:
                print(f"{population} {group}: fewer than 10 participants", flush=True)
                continue
            missing = MISSING_TABLE_FEATURES.get(group, [])
            reliability = "same model sliced by posture; start-without-seat mapped to supine"
            if population == "normotensive":
                reliability += "; resting SBP <= 130 from Calibration start"
            if missing:
                reliability = "missing " + ", ".join(missing) + "; " + reliability
            labels = usable["pid"].to_numpy()
            observed = usable["delta_sbp"].to_numpy(dtype=float)
            posture = usable["posture"].to_numpy()
            matrices = {
                "ridge": usable[columns].to_numpy(dtype=float),
                "static": usable[STATIC_FEATURES].to_numpy(dtype=float),
            }
            summaries = {}
            for model, matrix in (("ridge", matrices["ridge"]), ("static", matrices["static"]), ("zero", None)):
                predicted = np.zeros(observed.shape) if model == "zero" else lopo_predict(matrix, observed, labels)
                errors = predicted - observed
                full = {name: _slice_summary(errors[posture == name] if name != "all" else errors) for name in ("all", "supine", "seated")}
                collected = {name: {"bias": [], "sd": []} for name in ("all", "supine", "seated")}
                for chosen in _bootstrap_ids(np.unique(labels), replicates, BOOTSTRAP_FRACTION, seed):
                    chosen_rows = np.isin(labels, chosen)
                    if int(np.unique(labels[chosen_rows]).size) < 10:
                        continue
                    if model == "zero":
                        fold_prediction = np.zeros(int(chosen_rows.sum()))
                    else:
                        fold_prediction = lopo_predict(matrix[chosen_rows], observed[chosen_rows], labels[chosen_rows])
                    fold_errors = fold_prediction - observed[chosen_rows]
                    fold_posture = posture[chosen_rows]
                    for name in ("all", "supine", "seated"):
                        piece = fold_errors if name == "all" else fold_errors[fold_posture == name]
                        if piece.size < 2:
                            continue
                        collected[name]["bias"].append(float(piece.mean()))
                        collected[name]["sd"].append(float(piece.std(ddof=1)))
                summaries[model] = (full, collected)
            for model, (full, collected) in summaries.items():
                for name in ("all", "supine", "seated"):
                    bias_array = np.asarray(collected[name]["bias"], dtype=float)
                    sd_array = np.asarray(collected[name]["sd"], dtype=float)
                    rows.append(
                        {
                            "population": population,
                            "row_set": group,
                            "model": model,
                            "posture": name,
                            "target": "delta_sbp",
                            "n_rows": full[name]["n"],
                            "n_participants": int(usable.loc[usable["posture"].eq(name) if name != "all" else slice(None), "pid"].nunique()),
                            "lopo_bias": full[name]["bias"],
                            "lopo_sd": full[name]["sd"],
                            "boot_bias": float(bias_array.mean()) if bias_array.size else np.nan,
                            "boot_sd": float(sd_array.mean()) if sd_array.size else np.nan,
                            "boot_replicates_used": int(sd_array.size),
                            "missing_table_features": ",".join(missing),
                            "reliability": reliability,
                        }
                    )
            ridge_all = next(row for row in rows if row["population"] == population and row["row_set"] == group and row["model"] == "ridge" and row["posture"] == "all")
            ridge_supine = next(row for row in rows if row["population"] == population and row["row_set"] == group and row["model"] == "ridge" and row["posture"] == "supine")
            ridge_seated = next(row for row in rows if row["population"] == population and row["row_set"] == group and row["model"] == "ridge" and row["posture"] == "seated")
            print(
                f"{population} {group}: "
                f"all {ridge_all['boot_sd']:.2f}  supine {ridge_supine['boot_sd']:.2f}  seated {ridge_seated['boot_sd']:.2f}",
                flush=True,
            )
            if output_path:
                pd.DataFrame(rows).to_json(output_path, orient="records", indent=2)
    return pd.DataFrame(rows)
