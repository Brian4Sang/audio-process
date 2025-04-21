import subprocess as sp
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.stage_logger import StageLogger
from fish_audio_preprocess.utils.file import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    list_files,
    make_dirs,
)

"""
📦 convert_to_wav.py

功能：将音频/视频批量转换为 .wav 文件，支持自动切片、保留目录结构等。

🔧 使用：
    python tools/convert_to_wav.py [INPUT_DIR] [OUTPUT_DIR] [OPTIONS]

参数说明：
- input_dir        输入目录，支持音/视频文件及多级子目录
- output_dir       输出目录，生成的 .wav 文件将保存在此
- --recursive      是否递归查找子目录（默认开启）
- --overwrite      是否覆盖已存在的输出文件
- --clean          是否清空输出目录（⚠️ 谨慎使用）
- --segment N      将音频每 N 秒切一段(默认0 表示不切)
- --log-dir        保存处理过程的信息(default:None)

🎯 示例：
- 基本用法：
    python convert_to_wav.py data/raw data/wav
- 每段最多 5 分钟，递归处理：
    python convert_to_wav.py data/raw data/wav --segment 300
- 覆盖旧文件并清空输出目录：
    python convert_to_wav.py data/raw data/wav --overwrite --clean
"""


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files")
@click.option("--clean/--no-clean", default=False, help="Clean output directory before processing")
@click.option(
    "--segment",
    help="Maximum segment length in seconds, use 0 to disable",
    default=0,
    show_default=True,
)
@click.option(
    "--logdir",
    type=click.Path(file_okay=False),
    help="Directory to save processing logs (converted/skipped/failed files)",
    default=None,
)
def to_wav(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    segment: int,
    logdir: str,
):
    """Converts all audio and video files in input_dir to wav files in %output_dir%"""

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(
        input_dir, extensions=VIDEO_EXTENSIONS | AUDIO_EXTENSIONS, recursive=recursive
    )
    logger.info(f"Found {len(files)} files, converting to wav")

    stageLogger = StageLogger("convert_to_wav", logdir) if logdir else None
    skipped = 0

    for file in tqdm(files):
        relative_path = file.relative_to(input_dir)
        new_file = (
            output_dir
            / relative_path.parent
            / relative_path.name.replace(
                file.suffix, "_%04d.wav" if segment > 0 else ".wav"
            )
        )

        if stageLogger:
            stageLogger.log_total()

        if new_file.parent.exists() is False:
            new_file.parent.mkdir(parents=True)

        check_path = (
            (new_file.parent / (new_file.name % 0)) if segment > 0 else new_file
        )
        if check_path.exists() and not overwrite:
            skipped += 1
            if stageLogger:
                stageLogger.log_skipped(str(file), reason="文件已存在，未开启 overwrite")
            continue

        command = ["ffmpeg", "-i", str(file)]
        if segment > 0:
            command.extend(["-f", "segment", "-segment_time", str(segment)])
        command.append(str(new_file))

        try:
            sp.check_call(command, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
            if stageLogger:
                stageLogger.log_success()
        except Exception as e:
            logger.warning(f"Failed to convert {file}: {e}")
            if stageLogger:
                stageLogger.log_failed(str(relative_path), str(e))
            continue

    logger.info("Done!")
    logger.info(f"Total: {len(files)}, Skipped: {skipped}")
    logger.info(f"Output directory: {output_dir}")

    if stageLogger:
        stageLogger.save()


if __name__ == "__main__":
    to_wav()
