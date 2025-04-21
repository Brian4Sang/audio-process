#!/usr/bin/env python3
"""
响度归一化（Loudness Normalization）

功能：将所有音频响度统一到目标 LUFS 电平（ITU-R BS.1770-4），支持 peak 归一化。

参数说明：
- input_dir       输入音频文件夹（支持递归）
- output_dir      输出处理后的音频文件夹
- --loudness      目标响度（LUFS），默认 -23.0
- --peak          峰值归一化目标（dBFS），默认 -1.0
- --block-size    响度计算窗口大小（秒），默认 0.4s
- --recursive     是否递归处理子目录
- --overwrite     是否覆盖已有文件
- --clean         是否清空输出目录
- --logdir        日志输出目录
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
from loguru import logger
from tqdm import tqdm
import soundfile as sf
import pyloudnorm as pyln

from fish_audio_preprocess.utils.file import list_files, AUDIO_EXTENSIONS, make_dirs
from fish_audio_preprocess.utils.stage_logger import StageLogger


def normalize_file(input_file, output_file, loudness, peak, block_size, logger_obj=None):
    try:
        audio, rate = sf.read(input_file)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # 转为单通道

        # 峰值归一化
        audio = pyln.normalize.peak(audio, peak)

        meter = pyln.Meter(rate, block_size=block_size)
        loud = meter.integrated_loudness(audio)

        norm_audio = pyln.normalize.loudness(audio, loud, loudness)

        sf.write(output_file, norm_audio, rate)

        if logger_obj:
            logger_obj.log_success()

    except Exception as e:
        logger.warning(f"❌ Failed: {input_file} - {e}")
        if logger_obj:
            logger_obj.log_failed(str(input_file), reason=str(e))


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--loudness", default=-23.0, show_default=True, type=float, help="目标响度 (LUFS)")
@click.option("--peak", default=-1.0, show_default=True, type=float, help="峰值归一化 (dBFS)")
@click.option("--block-size", default=0.4, show_default=True, type=float, help="响度测量窗口大小（秒）")
@click.option("--recursive/--no-recursive", default=True, help="是否递归子目录")
@click.option("--overwrite/--no-overwrite", default=False, help="是否覆盖已有输出文件")
@click.option("--clean/--no-clean", default=False, help="是否清空输出目录")
@click.option("--num-workers", default=os.cpu_count(), show_default=True, type=int, help="并行处理进程数")
@click.option("--logdir", default="logs", type=click.Path(file_okay=False), help="日志输出目录")
def loudness_normalize(
    input_dir,
    output_dir,
    loudness,
    peak,
    block_size,
    recursive,
    overwrite,
    clean,
    num_workers,
    logdir,
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} files to normalize")

    stage_logger = StageLogger("loudness_normalize", logdir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for file in tqdm(files, desc="Preparing"):
            stage_logger.log_total()

            rel_path = file.relative_to(input_dir)
            out_file = output_dir / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            if out_file.exists() and not overwrite:
                stage_logger.log_skipped(str(file), reason="已存在，未开启 overwrite")
                continue

            futures.append(executor.submit(
                normalize_file,
                str(file),
                str(out_file),
                loudness,
                peak,
                block_size,
                stage_logger
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            _ = future.result()

    logger.info("✅ Loudness normalization completed")
    logger.info(f"Output directory: {output_dir}")
    stage_logger.save()


if __name__ == "__main__":
    loudness_normalize()
