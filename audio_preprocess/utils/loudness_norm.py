from pathlib import Path
from typing import Union

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def detect_and_fix_spikes(audio: np.ndarray, sample_rate: int, threshold_db: float = -3.0, window_ms: float = 3.0) -> np.ndarray:
    """
    Detect and smooth short-time abnormal peaks (spikes) in audio.

    Args:
        audio: numpy audio array (mono or stereo)
        sample_rate: sample rate in Hz
        threshold_db: spike detection threshold in dBFS (e.g., -3.0 dBFS)
        window_ms: window size in milliseconds for local analysis

    Returns:
        Processed audio with smoothed spikes
    """

    threshold_amp = 10 ** (threshold_db / 20)
    window_len = int(sample_rate * window_ms / 1000)
    if window_len < 1:
        return audio

    is_mono = audio.ndim == 1
    if is_mono:
        audio = audio[np.newaxis, :]  # shape -> (1, N)

    audio_fixed = np.copy(audio)
    num_fixed = 0

    for ch in range(audio.shape[0]):
        x = audio[ch]
        for i in range(window_len, len(x) - window_len):
            local = x[i - window_len:i + window_len + 1]
            local_median = np.median(local)
            local_std = np.std(local)

            # 条件放宽：移除局部突变点
            if np.abs(x[i]) > threshold_amp and np.abs(x[i]) > local_median + 2.5 * local_std:
                audio_fixed[ch, i] = local_median
                num_fixed += 1

    if num_fixed > 0:
        print(f"✅ 修复异常峰值: {num_fixed} 个点（>{threshold_db} dBFS）")

    return audio_fixed[0] if is_mono else audio_fixed


def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio array.
    """
    # Peak normalize
    audio = pyln.normalize.peak(audio, peak)

    # Loudness normalize
    meter = pyln.Meter(rate, block_size=block_size)
    current_loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, current_loudness, loudness)


def loudness_norm_file(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    peak=-3.0,
    loudness=-17.0,
    block_size=0.400,
) -> None:
    """
    Loudness normalization pipeline with spike repair.
    """
    input_file, output_file = str(input_file), str(output_file)

    audio, rate = sf.read(input_file)

    # Step 1: Spike Detection + Fix
    audio = detect_and_fix_spikes(audio, rate, threshold_db=peak)

    # Step 2: Loudness normalization
    meter = pyln.Meter(rate, block_size=block_size)
    before_loudness = meter.integrated_loudness(audio)

    audio = pyln.normalize.peak(audio, peak)
    audio = pyln.normalize.loudness(audio, before_loudness, loudness)

    after_loudness = meter.integrated_loudness(audio)

    print(f"{Path(input_file).name} | before: {before_loudness:.2f} LUFS → after: {after_loudness:.2f} LUFS")

    # Step 3: Save output
    sf.write(output_file, audio, rate)
