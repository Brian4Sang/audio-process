#!/usr/bin/env python3
"""
使用静音检测将长音频切割成多个短句

参数说明：
--min-duration         每段最短时长（秒）
--max-duration         每段最长时长（秒）
--min-silence-duration 静音段判定最小时长（秒）
--top-db               静音判断的能量阈值（dB）
--merge-short          是否将过短片段合并
--flat-layout          是否平铺输出所有切片
--logdir               保存处理日志的目录（默认 logs/）

输入：一个包含 `.wav` 的文件夹
输出：多个切片音频，命名为 xxx_0000.wav 形式
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs
from fish_audio_preprocess.utils.stage_logger import StageLogger


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="是否递归处理子目录")
@click.option("--overwrite/--no-overwrite", default=False, help="是否覆盖已有切片结果")
@click.option("--clean/--no-clean", default=False, help="是否清空输出目录")
@click.option("--num-workers", default=os.cpu_count(), type=int, show_default=True, help="并行处理的进程数")
@click.option("--min-duration", default=3.0, type=float, show_default=True, help="最小切片时长（秒）")
@click.option("--max-duration", default=12.0, type=float, show_default=True, help="最大切片时长（秒）")
@click.option("--min-silence-duration", default=0.3, type=float, show_default=True, help="静音最小持续时间（秒）")
@click.option("--top-db", default=-40, type=int, show_default=True, help="静音检测能量阈值（越小越严格）")
@click.option("--hop-length", default=10, type=int, show_default=True, help="静音检测跳帧长度")
@click.option("--max-silence-kept", default=0.5, type=float, show_default=True, help="每段保留静音最大长度（秒）")
@click.option("--flat-layout/--no-flat-layout", default=False, help="是否平铺所有输出切片")
@click.option("--merge-short/--no-merge-short", default=True, help="是否自动合并过短片段")
@click.option("--logdir", default="logs", type=click.Path(file_okay=False), help="保存日志的目录")
def slice_audio(
    input_dir, output_dir, recursive, overwrite, clean,
    num_workers, min_duration, max_duration, min_silence_duration,
    top_db, hop_length, max_silence_kept, flat_layout, merge_short, logdir
):
    from fish_audio_preprocess.utils.slice_audio_v2 import slice_audio_file_v2

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    make_dirs(output_dir, clean)

    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"Found {len(files)} audio files to process...")

    stage_logger = StageLogger("slice_audio", logdir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            stage_logger.log_total()
            relative_path = file.relative_to(input_dir)
            base_output = output_dir / relative_path.stem if not flat_layout else output_dir

            if not overwrite and base_output.exists():
                stage_logger.log_skipped(str(file), reason="目标已存在，未开启 overwrite")
                continue

            base_output.mkdir(parents=True, exist_ok=True)

            tasks.append(executor.submit(
                slice_audio_file_v2,
                input_file=str(file),
                output_dir=base_output,
                min_duration=min_duration,
                max_duration=max_duration,
                min_silence_duration=min_silence_duration,
                top_db=top_db,
                hop_length=hop_length,
                max_silence_kept=max_silence_kept,
                flat_layout=flat_layout,
                merge_short=merge_short,
            ))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Slicing"):
            try:
                future.result()
                stage_logger.log_success()
            except Exception as e:
                logger.warning(f"❌ Slice failed: {e}")
                stage_logger.log_failed(str(future), reason=str(e))

    stage_logger.save()
    logger.info(f"✅ Slice complete. Logs saved to {logdir}/slice_audio_log.json")


if __name__ == "__main__":
    slice_audio()
