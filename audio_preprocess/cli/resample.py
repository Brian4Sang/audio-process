import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs
from audio_preprocess.utils.stage_logger import StageLogger


def resample_file(
    input_file: Path,
    output_file: Path,
    overwrite: bool,
    target_sr: int,
    mono: bool,
    logger_obj: StageLogger = None,
    subtype: str = "PCM_16"
):
    import soundfile as sf
    import librosa

    try:
        if not input_file.exists():
            if logger_obj:
                logger_obj.log_skipped(str(input_file), reason="文件不存在")
            return

        # 获取原始采样率
        info = sf.info(str(input_file))
        if info.samplerate < target_sr:
            if logger_obj:
                logger_obj.log_skipped(str(input_file), reason=f"采样率过低: {info.samplerate}Hz")
            return

        if not overwrite and output_file.exists():
            if logger_obj:
                logger_obj.log_skipped(str(input_file), reason="输出文件已存在")
            return

        audio, _ = librosa.load(str(input_file), sr=target_sr, mono=mono)
        if audio.ndim == 2:
            audio = audio.T

        sf.write(str(output_file), audio, target_sr, subtype=subtype)
        if logger_obj:
            logger_obj.log_success()

    except Exception as e:
        if logger_obj:
            logger_obj.log_failed(str(input_file), reason=str(e))


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files")
@click.option("--clean/--no-clean", default=False, help="Clean output directory before processing")
@click.option("--num-workers", default=os.cpu_count(), show_default=True, type=int, help="Number of parallel workers")
@click.option("--sampling-rate", "-sr", default=24000, show_default=True, type=int, help="Target sampling rate")
@click.option("--mono/--nomo", default=True, help="Convert to mono")
@click.option("--log-dir", type=click.Path(file_okay=False), default=None, help="Directory to save processing logs")
@click.option(
    "--subtype",
    type=click.Choice(["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"], case_sensitive=False),
    default="PCM_16",
    show_default=True,
    help="Subtype to save audio format, e.g., PCM_16, FLOAT, etc."
)

def resample(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    num_workers: int,
    sampling_rate: int,
    mono: bool,
    log_dir: str,
    subtype: str
):
    """
    Resample all audio files in input_dir to output_dir.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, resampling to {sampling_rate} Hz")
    logger.info(f"Resampling to {sampling_rate} Hz, format={subtype}, mono={mono}")

    stageLogger = StageLogger("resample", log_dir) if log_dir else None

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        for file in tqdm(files, desc="Preparing tasks"):
            relative_path = file.relative_to(input_dir)
            new_file = output_dir / relative_path
            new_file.parent.mkdir(parents=True, exist_ok=True)

            if stageLogger:
                stageLogger.log_total()

            tasks.append(
                executor.submit(
                    resample_file,
                    file,
                    new_file,
                    overwrite,
                    sampling_rate,
                    mono,
                    stageLogger,
                    subtype
                )
            )

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            exception = future.exception()
            if exception:
                logger.error(f"Exception in worker: {exception}")

    logger.info("Done!")
    logger.info(f"Total: {len(files)}")
    logger.info(f"Output directory: {output_dir}")

    if stageLogger:
        stageLogger.save()


if __name__ == "__main__":
    resample()