#!/usr/bin/env python3
"""
人声分离基于 Demucs

功能：将包含背景音乐或混合信号的音频进行分离，只保留指定音轨（如 vocals）。

可配置参数：
- input_dir          输入音频目录（.wav）
- output_dir         输出分离后音频目录
- --track            要保留的音轨名（默认为 vocals）
- --model            使用的分离模型（如 htdemucs）
- --overwrite        是否覆盖已有输出文件
- --logdir           保存处理日志的目录（使用 StageLogger）
- --recursive        是否递归查找子目录（默认开启）
- --num_workers_per_gpu 并发进程数（多 GPU）
- --strict-check     是否跳过不符合Demucs要求的音频
"""

#!/usr/bin/env python3
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING
import json

import click
from loguru import logger
from tqdm import tqdm
import soundfile as sf

from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs
from fish_audio_preprocess.utils.stage_logger import StageLogger

if TYPE_CHECKING:
    import torch


def worker(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    track: list[str],
    model: str,
    shifts: int,
    device: "torch.device",
    shard_idx: int = -1,
    total_shards: int = 1,
    logdir: str = None,
    strict_check: bool = True,
):
    from fish_audio_preprocess.utils.separate_audio import (
        init_model,
        load_track,
        merge_tracks,
        save_audio,
        separate_audio,
    )

    logger_obj = StageLogger(f"separate_shard{shard_idx}", logdir) if logdir else None
    files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    if shard_idx >= 0:
        files = [f for i, f in enumerate(files) if i % total_shards == shard_idx]

    shard_name = f"[Shard {shard_idx + 1}/{total_shards}]"
    logger.info(f"{shard_name} Found {len(files)} files, separating audio")

    _model = init_model(model, device)

    for file in tqdm(
        files,
        desc=f"{shard_name} Separating audio",
        position=0 if shard_idx < 0 else shard_idx,
        leave=False,
    ):
        if logger_obj:
            logger_obj.log_total()

        relative_path = file.relative_to(input_dir)
        new_file = output_dir / relative_path

        if new_file.exists() and not overwrite:
            if logger_obj:
                logger_obj.log_skipped(str(file), reason="已存在,未设置 overwrite")
            continue

        try:
            info = sf.info(str(file))
            if strict_check:
                if info.samplerate not in [44100, 48000] or info.channels != 2:
                    reason = f"当前音频: {info.samplerate}Hz, {info.channels}ch,不满足 Demucs 要求"
                    logger.warning(f"⚠️ 跳过: {file} - {reason}")
                    if logger_obj:
                        logger_obj.log_skipped(str(file), reason=reason)
                    continue
        except Exception as e:
            logger.warning(f"⚠️ 无法读取音频信息 {file}: {e}")
            if logger_obj:
                logger_obj.log_failed(str(file), reason=f"音频信息读取失败: {e}")
            continue

        try:
            source = load_track(_model, file)
            separated = separate_audio(_model, source, shifts=shifts, num_workers=0)
            merged = merge_tracks(separated, track)
            save_audio(_model, new_file, merged)
            if logger_obj:
                logger_obj.log_success()
        except Exception as e:
            logger.warning(f"❌ Failed to separate {file}: {e}")
            if logger_obj:
                logger_obj.log_failed(str(file), reason=str(e))

    if logger_obj:
        logger_obj.save()

    logger.info(f"{shard_name} Finished.")


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite existing files")
@click.option("--clean/--no-clean", default=False, help="Clean output directory before processing")
@click.option("--track", "-t", multiple=True, help="Name of track(s) to keep", default=["vocals"])
@click.option("--model", help="Name of model to use", default="htdemucs")
@click.option("--shifts", help="Number of shifts, improves separation quality a bit", default=1)
@click.option("--num_workers_per_gpu", help="Number of workers per GPU", default=2)
@click.option("--logdir", type=click.Path(file_okay=False), default=None, help="Directory to save processing logs")
@click.option(
    "--strict-check/--no-strict-check",
    default=True,
    show_default=True,
    help="是否跳过采样率或通道数不满足 Demucs 要求的音频（44100/48000Hz且为双声道）"
)
def separate(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    track: list[str],
    model: str,
    shifts: int,
    num_workers_per_gpu: int,
    logdir: str,
    strict_check: bool,
):
    """
    Separates audio in input_dir using model and saves to output_dir.
    """

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir == output_dir and clean:
        logger.error("You are trying to clean the input directory, aborting")
        return

    make_dirs(output_dir, clean)

    base_args = (
        input_dir,
        output_dir,
        recursive,
        overwrite,
        track,
        model,
        shifts,
    )

    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        logger.info(f"Using {torch.cuda.device_count()} GPU(s)...")
        mp.set_start_method("spawn", force=True)
        processes = []
        shards = torch.cuda.device_count() * num_workers_per_gpu
        for shard_idx in range(shards):
            p = mp.Process(
                target=worker,
                args=(
                    *base_args,
                    torch.device(f"cuda:{shard_idx % torch.cuda.device_count()}"),
                    shard_idx,
                    shards,
                    logdir,
                    strict_check,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        logger.info("⚠️ CUDA not available. Falling back to CPU...")
        worker(
            *base_args,
            torch.device("cpu"),
            shard_idx=-1,
            total_shards=1,
            logdir=logdir,
            strict_check=strict_check,
        )

    # ✅ 合并日志
    if logdir:
        merged_file, summary = merge_stage_logs(logdir)
        logger.info(f"✅ 合并日志完成：{logdir}/{merged_file}")
        logger.info(f"📊 日志汇总: {summary}")


# ✅ 附加日志合并工具
def merge_stage_logs(logdir: str, prefix: str = "separate_shard", merged_name: str = "separate_log.json"):
    logdir = Path(logdir)
    merged = {
        "stage": "separate",
        "summary": {"total": 0, "success": 0, "skipped": 0, "failed": 0},
        "skipped": [],
        "failed": []
    }

    shard_logs = list(logdir.glob(f"{prefix}*_log.json"))

    for log_file in logdir.glob(f"{prefix}*_log.json"):
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged["summary"]["total"] += data["summary"].get("total", 0)
            merged["summary"]["success"] += data["summary"].get("success", 0)
            merged["summary"]["skipped"] += data["summary"].get("skipped", 0)
            merged["summary"]["failed"] += data["summary"].get("failed", 0)
            merged["skipped"].extend(data.get("skipped", []))
            merged["failed"].extend(data.get("failed", []))

    merged_path = logdir / merged_name
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # ✅ 删除单独的 shard 日志
    for log_file in shard_logs:
        log_file.unlink()

    return merged_path.name, merged["summary"]


if __name__ == "__main__":
    separate()
