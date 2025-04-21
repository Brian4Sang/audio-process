
#!/usr/bin/env python3
"""
多说话人检测（Multi-speaker Detection）

功能：分析每条语音是否包含多位说话人，通过提取 speaker embedding 并聚类判断。

输入：
- 一个音频目录，包含多个已切片的 wav 文件（如切片结果）

输出：
- 一个 JSON 文件，记录每条语音是否为多说话人语音
- 可选输出：过滤后的音频文件夹（只保留单说话人）

参数：
--input-dir         输入音频目录
--output-json       保存检测结果的路径（默认 multi_spk_result.json）
--embedding-model   用于提取 speaker embedding 的模型路径（onnx）
--threshold         判断为多说话人的类间距离阈值（默认 0.75）
--logdir            保存处理日志的目录（默认 logs/）
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import click
from loguru import logger

from fish_audio_preprocess.utils.stage_logger import StageLogger
from fish_audio_preprocess.utils.file import list_files, AUDIO_EXTENSIONS
from fish_audio_preprocess.utils.extract_embedding import extract_embedding
from fish_audio_preprocess.utils.cluster_embeddings import cluster_embeddings



@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), required=True, help="输入音频目录")
@click.option("--embedding-model", type=click.Path(exists=True, dir_okay=False), required=True, help="Speaker embedding 模型 (ONNX)")
@click.option("--output-json", type=click.Path(), default="multi_spk_result.json", show_default=True, help="多说话人检测结果")
@click.option("--threshold", type=float, default=0.75, show_default=True, help="聚类距离阈值")
@click.option("--logdir", type=click.Path(), default="logs", show_default=True, help="保存日志目录")
def detect_multi_spk(input_dir, embedding_model, output_json, threshold, logdir):
    input_dir = Path(input_dir)
    all_wavs = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=True)
    logger.info(f"🔍 Found {len(all_wavs)} wavs for speaker detection")

    result = {}
    stage_logger = StageLogger("multi_spk_detect", logdir)

    for wav_path in tqdm(all_wavs):
        stage_logger.log_total()
        try:
            utt_id = str(wav_path.relative_to(input_dir)).replace("/", "_")
            embeddings = extract_embedding(str(wav_path), embedding_model)
            label_count = cluster_embeddings(embeddings, threshold)
            result[utt_id] = {"multi_speaker": label_count > 1, "spk_count": label_count}
            stage_logger.log_success()
        except Exception as e:
            logger.warning(f"❌ Failed on {wav_path}: {e}")
            stage_logger.log_failed(str(wav_path), str(e))

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    stage_logger.save()
    logger.info(f"✅ Detection result saved to {output_json}")


if __name__ == "__main__":
    detect_multi_spk()
