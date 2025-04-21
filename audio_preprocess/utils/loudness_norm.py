from pathlib import Path
from typing import Union

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        audio: audio data
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)

    Returns:
        loudness normalized audio
    """

    # peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)

    # measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    return pyln.normalize.loudness(audio, _loudness, loudness)


def loudness_norm_file(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    peak=-1.0,
    loudness=-23.0,
    block_size=0.400,
) -> None:
    import pyloudnorm as pyln

    input_file, output_file = str(input_file), str(output_file)

    audio, rate = sf.read(input_file)

    meter = pyln.Meter(rate, block_size=block_size)
    before_loudness = meter.integrated_loudness(audio)

    audio = pyln.normalize.peak(audio, peak)
    audio = pyln.normalize.loudness(audio, before_loudness, loudness)

    after_loudness = meter.integrated_loudness(audio)
    print(f"{Path(input_file).name} | before: {before_loudness:.2f} LUFS → after: {after_loudness:.2f} LUFS")

    sf.write(output_file, audio, rate)
