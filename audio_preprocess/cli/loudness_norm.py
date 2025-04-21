#!/usr/bin/env python3
"""
Stage 6: Loudness normalization（响度归一化）

功能：
  使用 ITU-R BS.1770-4 标准对音频响度进行归一化（LUFS + Peak），提升整体响度一致性。

输入：
  - 一个包含多个 .wav 文件的目录（支持递归）

输出：
  - 一个响度归一化后的音频目录（结构与输入一致）

可配置参数：
  --peak             峰值归一化（单位 dB，默认 -1.0）
  --loudness         响度目标值（单位 LUFS，默认 -23.0）
  --block-size       响度计算窗口（秒，默认 0.4）
  --logdir           日志保存目录（如 logs/）
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs
from fish_audio_preprocess.utils.loudness_norm import loudness_norm_file
from fish_audio_preprocess.utils.stage_logger import StageLogger


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files")
@click.option("--clean/--no-clean", default=False, help="Clean output directory before processing")
@click.option("--peak", default=-1.0, type=float, show_default=True, help="Peak normalize audio to -1 dB")
@click.option("--loudness", default=-23.0, type=float, show_default=True, help="Loudness normalize audio to -23 dB LUFS")
@click.option("--block-size", default=0.4, type=float, show_default=True, help="Block size for loudness measurement, unit is second")
@click.option("--num-workers", default=os.cpu_count(), type=int, show_default=True, help="Number of parallel workers")
@click.option("--logdir", type=click.Path(file_okay=False), default=None, help="Directory to save processing logs")
def loudness_norm(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    peak: float,
    loudness: float,
    block_size: float,
    num_workers: int,
    logdir: str,
):
    """Perform loudness normalization (ITU-R BS.1770-4) on audio files."""

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files, normalizing loudness...")

    stage_logger = StageLogger("loudness_normalize", logdir) if logdir else None

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            relative_path = file.relative_to(input_dir)
            new_file = output_dir / relative_path

            if stage_logger:
                stage_logger.log_total()

            if new_file.exists() and not overwrite:
                if stage_logger:
                    stage_logger.log_skipped(str(file), reason="文件已存在，未开启 overwrite")
                continue

            if new_file.parent.exists() is False:
                new_file.parent.mkdir(parents=True)

            tasks.append(
                executor.submit(
                    loudness_norm_file,
                    file,
                    new_file,
                    peak,
                    loudness,
                    block_size,
                )
            )

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            try:
                future.result()
                if stage_logger:
                    stage_logger.log_success()
            except Exception as e:
                logger.warning(f"❌ Failed to normalize: {e}")
                if stage_logger:
                    stage_logger.log_failed(str(future), reason=str(e))

    logger.info("✅ Done!")
    logger.info(f"🎧 Total files: {len(files)}")

    if stage_logger:
        stage_logger.save()
        logger.info(f"📄 Log saved to: {logdir}/loudness_normalize_log.json")


if __name__ == "__main__":
    loudness_norm()
