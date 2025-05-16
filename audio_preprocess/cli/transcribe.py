#!/usr/bin/env python3
"""
Stage 7: ASR 识别(音频转文本)

功能：对指定目录中的所有音频文件进行语音识别，输出对应的 .lab 文本(纯文本，无标点)。

支持模型：
- Whisper (faster-whisper)
- FunASR

输出：
- 每条音频同目录生成一个 .lab 文件
- 日志文件记录识别情况(成功 / 跳过 / 失败)

参数：
--lang             语言代码(如 zh, en)
--model-type       模型类型(zh:paraformer-zh and sensevoice-small) (en:whisper and paraformer-en)
--compute-type     Whisper 的计算类型: float16 / float32 / int8
--batch-size       Whisper 的 batch size(仅在支持 batched 推理时生效)
--logdir           日志保存目录(默认 logs/)
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import torch
from loguru import logger
from tqdm import tqdm

from audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, split_list
from audio_preprocess.utils.stage_logger import StageLogger
from audio_preprocess.utils.transcribe import ASRModelType, batch_transcribe


def replace_lastest(string, old, new):
    return string[::-1].replace(old[::-1], new[::-1], 1)[::-1]


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--num-workers", default=2, show_default=True, type=int, help="并行处理的进程数")
@click.option("--lang", default="zh", show_default=True, help="识别语言代码")
@click.option("--model-type", default="paraformer", show_default=True, help="模型类型：whisper 或 funasr")
@click.option("--compute-type", default="float16", show_default=True, help="Whisper的推理精度")
@click.option("--batch-size", default=1, show_default=True, help=" batch size(仅batched支持)")
@click.option("--recursive/--no-recursive", default=True, help="是否递归扫描子目录")
@click.option("--logdir", default="logs", type=click.Path(file_okay=False), help="日志保存目录")
def transcribe(
    input_dir: str,
    num_workers: int,
    lang: str,
    # model_size: str,
    model_type: ASRModelType,
    compute_type: str,
    batch_size: int,
    recursive: bool,
    logdir: str,
):
    """对目录下所有音频文件进行语音识别，输出 .lab 文件"""

    ctx = click.get_current_context()
    provided_options = {
        key: value
        for key, value in ctx.params.items()
        if ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE
    }

    if model_type == "funasr" :
        logger.info("未指定 funasr 模型，使用默认模型：iic/SenseVoiceSmall")
        model_size = "iic/SenseVoiceSmall"

    if not torch.cuda.is_available():
        logger.warning("检测CUDA不可用，将使用 CPU 进行识别")

    logger.info(f"识别目录：{input_dir}")
    logger.info(f"模型类型：{model_type} |  语言：{lang}")
    logger.info(f"并行进程数：{num_workers}")

    audio_files = list_files(input_dir, extensions=AUDIO_EXTENSIONS, recursive=recursive)
    audio_files = [str(f) for f in audio_files]

    if len(audio_files) == 0:
        logger.error("未找到音频文件，退出")
        return

    chunks = split_list(audio_files, num_workers)
    stage_logger = StageLogger("asr_transcribe", logdir)

    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
        tasks = []
        for chunk_id, chunk in enumerate(chunks):
            tasks.append(
                executor.submit(
                    batch_transcribe,
                    files=chunk,
                    # model_size=model_size,
                    model_type=model_type,
                    lang=lang,
                    pos=chunk_id,
                    compute_type=compute_type,
                    batch_size=batch_size,
                )
            )

        results = {}
        for task in tasks:
            try:
                results.update(task.result())
            except Exception as e:
                logger.warning(f"子进程识别失败：{e}")

    logger.info(" 开始保存 .lab 文本")

    for wav_path in tqdm(results.keys()):
        stage_logger.log_total()
        try:
            lab_path = replace_lastest(wav_path, ".wav", ".lab")
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(results[wav_path])
            stage_logger.log_success()
        except Exception as e:
            stage_logger.log_failed(wav_path, reason=str(e))
            logger.warning(f"写入失败 {wav_path}: {e}")

    logger.info("音频转写完成")
    stage_logger.save()
    logger.info(f"日志保存至：{logdir}/asr_transcribe_log.json")


if __name__ == "__main__":
    transcribe()
