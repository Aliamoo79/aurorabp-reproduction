"""Filters from Mieloszyk et al., IEEE JBHI 2022, Section II-B.

Source labels in this module:
- [PAPER] stated by Mieloszyk et al.
- [DECISION] not stated by the paper or the official repository.

The official waveform notebook uses a different Butterworth bandpass and
labels that function ad hoc. Those coefficients are not used here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
from scipy import signal

# [DECISION] Released files are resampled to 500 Hz. The paper does not
# state the sampling rate used to design the Matlab filters.
RELEASED_FS_HZ = 500.0

# [DECISION] Zero-phase filtering. The paper does not say causal or zero-phase.
# sosfiltfilt squares the magnitude response, so passband ripple in dB doubles.
ZERO_PHASE = True


def _as_float(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("Each waveform must be a one-dimensional array.")
    return values


def apply_sos(sos: np.ndarray, values: np.ndarray, zero_phase: bool = ZERO_PHASE) -> np.ndarray:
    """Apply an IIR filter in second-order sections."""
    data = _as_float(values)
    if data.size == 0 or not np.isfinite(data).any():
        return np.full(data.shape, np.nan)
    finite = np.isfinite(data)
    if not finite.all():
        # [DECISION] interpolate only internal isolated gaps. Edges stay NaN.
        index = np.arange(data.size)
        data = data.copy()
        data[~finite] = np.interp(index[~finite], index[finite], data[finite])
    if zero_phase:
        return signal.sosfiltfilt(sos, data)
    return signal.sosfilt(sos, data)


def apply_fir(coefficients: np.ndarray, values: np.ndarray, zero_phase: bool = ZERO_PHASE) -> np.ndarray:
    """Apply an FIR filter. Zero-phase uses forward-backward filtering."""
    data = _as_float(values)
    if data.size == 0 or not np.isfinite(data).any():
        return np.full(data.shape, np.nan)
    if zero_phase:
        return signal.filtfilt(coefficients, [1.0], data)
    return signal.lfilter(coefficients, [1.0], data)


@lru_cache(maxsize=4)
def design_ecg(fs: float = RELEASED_FS_HZ) -> Dict[str, np.ndarray]:
    """ECG: DC block, elliptic lowpass, Chebyshev Type I notch. [PAPER]

    Unspecified pieces, all [DECISION]:
    - DC block is a first-order Butterworth highpass at 0.1 Hz.
    - Elliptic stopband attenuation is 40 dB. At the paper's edges and
      0.1 dB ripple, ellipord returns order 7 only near this attenuation.
    - Notch Q is center/bandwidth, so the order-6 bandstop spans 50–70 Hz.
      scipy.signal.cheby1 uses N=3 for a 6th-order bandstop.
    """
    dc = signal.butter(1, 0.1, btype="highpass", output="sos", fs=fs)
    lowpass = signal.ellip(7, 0.1, 40.0, 40.0, btype="lowpass", output="sos", fs=fs)
    bandwidth = 60.0 / 3.0
    edges = (60.0 - bandwidth / 2.0, 60.0 + bandwidth / 2.0)
    notch = signal.cheby1(3, 0.1, edges, btype="bandstop", output="sos", fs=fs)
    return {"dc": dc, "lowpass": lowpass, "notch": notch}


@lru_cache(maxsize=4)
def design_ppg(fs: float = RELEASED_FS_HZ) -> Dict[str, np.ndarray]:
    """PPG: Butterworth highpass and equiripple lowpass. [PAPER]

    The equiripple order is not stated. 521 taps is the shortest odd length
    found with scipy.signal.remez at 500 Hz that meets 1 dB and 60 dB in a
    single pass. [DECISION]
    """
    if abs(fs - RELEASED_FS_HZ) > 1e-6:
        raise ValueError("The cached equiripple length is valid only at 500 Hz. Redesign it for another rate.")
    highpass = signal.butter(4, 0.25, btype="highpass", output="sos", fs=fs)
    ripple_db = 1.0
    attenuation_db = 60.0
    deviation_pass = (10 ** (ripple_db / 20.0) - 1.0) / (10 ** (ripple_db / 20.0) + 1.0)
    deviation_stop = 10 ** (-attenuation_db / 20.0)
    lowpass = signal.remez(
        521,
        [0.0, 10.0, 12.0, fs / 2.0],
        [1.0, 0.0],
        weight=[1.0 / deviation_pass, 1.0 / deviation_stop],
        fs=fs,
        maxiter=50,
    )
    return {"highpass": highpass, "lowpass": lowpass}


@lru_cache(maxsize=4)
def design_tonometry(fs: float = RELEASED_FS_HZ) -> Dict[str, np.ndarray]:
    """Tonometry: elliptic highpass and 7th-order elliptic lowpass. [PAPER]

    [DECISION] Highpass order is 6, the minimum returned by ellipord for the
    stated edges, 1 dB ripple, and 60 dB attenuation.
    [DECISION] Lowpass stopband attenuation is 40 dB so that order 7 is
    consistent with a 22 Hz passband edge and a 26 Hz stopband edge.
    """
    highpass = signal.ellip(6, 1.0, 60.0, 0.3, btype="highpass", output="sos", fs=fs)
    lowpass = signal.ellip(7, 0.1, 40.0, 22.0, btype="lowpass", output="sos", fs=fs)
    return {"highpass": highpass, "lowpass": lowpass}


def filter_ecg(values: np.ndarray, fs: float = RELEASED_FS_HZ, zero_phase: bool = ZERO_PHASE) -> np.ndarray:
    """Apply the ECG filters in the order listed in Section II-B."""
    stages = design_ecg(fs)
    output = apply_sos(stages["dc"], values, zero_phase=zero_phase)
    output = apply_sos(stages["lowpass"], output, zero_phase=zero_phase)
    return apply_sos(stages["notch"], output, zero_phase=zero_phase)


def filter_ppg(values: np.ndarray, fs: float = RELEASED_FS_HZ, zero_phase: bool = ZERO_PHASE) -> np.ndarray:
    """Apply the PPG filters in the order listed in Section II-B."""
    stages = design_ppg(fs)
    output = apply_sos(stages["highpass"], values, zero_phase=zero_phase)
    return apply_fir(stages["lowpass"], output, zero_phase=zero_phase)


def filter_tonometry(values: np.ndarray, fs: float = RELEASED_FS_HZ, zero_phase: bool = ZERO_PHASE) -> np.ndarray:
    """Apply the tonometry filters in the order listed in Section II-B."""
    stages = design_tonometry(fs)
    output = apply_sos(stages["highpass"], values, zero_phase=zero_phase)
    return apply_sos(stages["lowpass"], output, zero_phase=zero_phase)


def gain_db_at(kind: str, stage: str, target_hz: float, fs: float = RELEASED_FS_HZ) -> float:
    """Gain at one exact frequency, not at the nearest FFT bin."""
    if kind == "ecg":
        coefficients = design_ecg(fs)[stage]
    elif kind == "ppg":
        coefficients = design_ppg(fs)[stage]
    elif kind == "tonometry":
        coefficients = design_tonometry(fs)[stage]
    else:
        raise ValueError(f"Unknown filter group: {kind}")
    if coefficients.ndim == 1:
        _, response = signal.freqz(coefficients, worN=[target_hz], fs=fs)
    else:
        _, response = signal.sosfreqz(coefficients, worN=[target_hz], fs=fs)
    return float(20.0 * np.log10(max(abs(response[0]), 1e-12)))


def stage_response(kind: str, stage: str, fs: float = RELEASED_FS_HZ, wor_n: int = 16384) -> Dict[str, np.ndarray]:
    """Frequency response of one designed stage, single pass."""
    if kind == "ecg":
        coefficients = design_ecg(fs)[stage]
    elif kind == "ppg":
        coefficients = design_ppg(fs)[stage]
    elif kind == "tonometry":
        coefficients = design_tonometry(fs)[stage]
    else:
        raise ValueError(f"Unknown filter group: {kind}")
    if coefficients.ndim == 1:
        frequencies, response = signal.freqz(coefficients, worN=wor_n, fs=fs)
    else:
        frequencies, response = signal.sosfreqz(coefficients, worN=wor_n, fs=fs)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    return {"hz": frequencies, "db": magnitude_db}


def specification_check(fs: float = RELEASED_FS_HZ) -> Dict[str, float]:
    """Measure single-pass gain at the paper's stated edges.

    Passband elliptic and Chebyshev edges are the -ripple points, not the
    -3 dB points. Butterworth edges are the -3 dB points.
    """
    probes = {
        "ecg_dc_0.1hz_db": ("ecg", "dc", 0.1),
        "ecg_lowpass_40hz_db": ("ecg", "lowpass", 40.0),
        "ecg_lowpass_45hz_db": ("ecg", "lowpass", 45.0),
        "ecg_notch_50hz_db": ("ecg", "notch", 50.0),
        "ecg_notch_60hz_db": ("ecg", "notch", 60.0),
        "ecg_notch_70hz_db": ("ecg", "notch", 70.0),
        "ppg_highpass_0.25hz_db": ("ppg", "highpass", 0.25),
        "ppg_lowpass_10hz_db": ("ppg", "lowpass", 10.0),
        "ppg_lowpass_12hz_db": ("ppg", "lowpass", 12.0),
        "tonometry_highpass_0.2hz_db": ("tonometry", "highpass", 0.2),
        "tonometry_highpass_0.3hz_db": ("tonometry", "highpass", 0.3),
        "tonometry_lowpass_22hz_db": ("tonometry", "lowpass", 22.0),
        "tonometry_lowpass_26hz_db": ("tonometry", "lowpass", 26.0),
    }
    return {name: gain_db_at(kind, stage, hz, fs=fs) for name, (kind, stage, hz) in probes.items()}
