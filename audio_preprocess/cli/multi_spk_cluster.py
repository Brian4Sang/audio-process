#!/usr/bin/env python3
"""
多说话人聚类划分（Speaker Clustering）

功能：提取每条语音的全局说话人 embedding，通过聚类将其分配到不同的说话人标签中，实现对无标签音频数据的说话人归类。

输入：
- 一个包含切片音频的目录，可包含多级子目录

输出：
- 一个 JSON 文件，记录每条语音对应的说话人标签（如 spk0, spk1...）
- 一个输出目录，按聚类标签组织音频文件（spk0/, spk1/...）

参数：
--input-dir           输入音频目录（可含子目录）
--embedding-model     用于提取 speaker embedding 的 ONNX 模型路径
--output-json         输出 JSON 路径（如 utt2spk.json）
--output-split-dir    按聚类输出的音频目录
--threshold           聚类距离阈值（默认 0.75）
--logdir              日志保存目录
--recursive           是否递归查找子目录中的音频（默认否）

--method:
dbscan         # 使用 DBSCAN 聚类（自动决定类数，适合“说话人未知”的情况）
kmeans         # 使用 KMeans（自动选类数k，但不支持 k=1）
agglomerative  # 使用层次聚类（需要指定阈值）

"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import click
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances

from audio_preprocess.utils.file import list_files, AUDIO_EXTENSIONS
from audio_preprocess.utils.extract_embedding import extract_embedding
from audio_preprocess.utils.stage_logger import StageLogger
from audio_preprocess.utils import cluster_embeddings as clustering_utils


@click.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True, help="输入音频目录")
@click.option("--embedding-model", type=click.Path(exists=True), required=True, help="Speaker embedding 模型路径 (ONNX)")
@click.option("--output-json", type=click.Path(), required=True, help="保存聚类结果 utt2spk.json")
@click.option("--output-split-dir", type=click.Path(), required=True, help="聚类后音频输出目录")
@click.option("--logdir", type=click.Path(), default="logs", show_default=True, help="日志输出目录")
@click.option("--recursive", is_flag=True, default=False, help="是否递归查找子目录")
@click.option("--method", type=click.Choice(["kmeans", "agglomerative", "dbscan"]), default="dbscan", show_default=True, help="聚类方法")
def main(input_dir, embedding_model, output_json, output_split_dir, logdir, recursive, method):
    input_dir = Path(input_dir)
    output_split_dir = Path(output_split_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    all_wavs = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    logger.info(f"共发现 {len(all_wavs)} 条音频")

    utt2embed = {}
    stage_logger = StageLogger("cluster_speakers", logdir)

    logger.info("开始提取 embedding")
    for wav in tqdm(all_wavs, desc="提取 embedding"):
        stage_logger.log_total()
        utt_id = str(wav.relative_to(input_dir)).replace("/", "_")
        try:
            emb = extract_embedding(str(wav), embedding_model)

                # ✅ 先判断 emb 是否为空或为 None
            if emb is None or len(emb) == 0:
                logger.warning(f"跳过 {wav}，提取的 embedding 为空")
                stage_logger.log_failed(str(wav), reason="empty embedding")
                continue

            mean_emb = emb.mean(axis=0)

            # ✅ 再判断 mean_emb 是否包含 NaN
            if np.isnan(mean_emb).any():
                logger.warning(f"跳过 {wav}，embedding 含 NaN")
                stage_logger.log_failed(str(wav), reason="embedding is NaN")
                continue

            utt2embed[utt_id] = emb.mean(axis=0).tolist()
            stage_logger.log_success()
        except Exception as e:
            logger.warning(f"❌ Failed on {wav}: {e}")
            stage_logger.log_failed(str(wav), str(e))

    logger.info(f"完成 embedding 提取，共计 {len(utt2embed)} 条")
    stage_logger.save()

    if len(utt2embed) == 0:
        logger.error("❌ 无可用 embedding，退出")
        return

    logger.info("计算 embedding 余弦距离统计")
    X = np.array(list(utt2embed.values()))
    cos_dists = cosine_distances(X)
    upper = cos_dists[np.triu_indices_from(cos_dists, k=1)]
    logger.info(f"余弦距离 - 平均: {upper.mean():.4f}, 最小: {upper.min():.4f}, 最大: {upper.max():.4f}")

    # 选择聚类方式
    logger.info(f"使用方法: {method} 开始聚类处理...")
    if method == "kmeans":
        labels, k = clustering_utils.cluster_embeddings_kmeans(X)
        logger.info(f"✅ 聚类完成，KMeans 最佳聚类数: {k}")
    elif method == "agglomerative":
        labels = clustering_utils.cluster_embeddings_agglomerative(X, threshold=0.75)
        logger.info(f"✅ 聚类完成，层次聚类得到 {len(set(labels))} 个类")
    elif method == "dbscan":
        labels = clustering_utils.cluster_embeddings_dbscan(X, eps=0.25)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"✅ 聚类完成，DBSCAN 得到 {n_clusters} 个聚类，标签: {set(labels)}")
    else:
        raise ValueError("不支持的聚类方法")

    utt_ids = list(utt2embed.keys())
    utt2spk = {utt: f"spk{label}" for utt, label in zip(utt_ids, labels)}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(utt2spk, f, indent=2, ensure_ascii=False)
    logger.info(f"聚类标签已保存至 {output_json}")

    logger.info(f"复制音频文件到 {output_split_dir}")
    for wav in all_wavs:
        utt_id = str(wav.relative_to(input_dir)).replace("/", "_")
        if utt_id not in utt2spk:
            continue
        spk_dir = output_split_dir / utt2spk[utt_id]
        spk_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(wav, spk_dir / wav.name)

    logger.info(f"✅ 完成。共检测出 {len(set(labels))} 个说话人，文件已组织输出。")


if __name__ == "__main__":
    main()
