#!/usr/bin/env python3
"""
DNSMOS 打分

功能：对目录下所有音频进行 DNSMOS 评估，输出质量分数（SIG, BAK, OVRL, P808_MOS）
支持 DNSMOS v8 模型和个性化模型，输出为 CSV，可用于筛选高质量语音。

参数：
--input-dir           输入音频目录
--output-csv          打分结果 CSV 文件路径
--personalized        是否启用个性化模型（pDNSMOS）
--logdir              日志目录（默认 logs/）
"""

import os
import glob
import click
import soundfile as sf
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path
from loguru import logger

from audio_preprocess.utils.stage_logger import StageLogger

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

def audio_melspec(audio, sr=16000, n_mels=120, frame_size=320, hop_length=160, to_db=True):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
    if to_db:
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
    return mel_spec.T

def get_polyfit_val(sig, bak, ovr, personalized):
    if personalized:
        p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
        p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
        p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
    else:
        p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
        p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
        p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
    return p_sig(sig), p_bak(bak), p_ovr(ovr)

@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), required=True, help="输入音频目录")
@click.option("--output-csv", type=click.Path(), default="dnsmos_result.csv", show_default=True, help="输出CSV路径")
@click.option("--personalized", is_flag=True, default=False, help="是否使用个性化模型（pDNSMOS）")
@click.option("--logdir", type=click.Path(), default="logs", show_default=True, help="日志输出目录")
def main(input_dir, output_csv, personalized, logdir):
    input_dir = Path(input_dir)
    all_wavs = [str(p) for p in input_dir.rglob("*.wav")]
    logger.info(f"共发现 {len(all_wavs)} 条音频")

    stage_logger = StageLogger("dnsmos", logdir)

    base_dir = Path("pretrained_models/dnsmos")

    for model_path in [
        base_dir / "DNSMOS/model_v8.onnx",
        base_dir / ("pDNSMOS/pdns_sig_bak_ovr.onnx" if personalized else "DNSMOS/sig_bak_ovr.onnx")
    ]:
        if not model_path.exists():
            raise FileNotFoundError(f"dnsmos模型文件未找到: {model_path}")

    p808_model = ort.InferenceSession(str(base_dir / "DNSMOS/model_v8.onnx"))
    main_model = ort.InferenceSession(str(
        base_dir / ("pDNSMOS/pdns_sig_bak_ovr.onnx" if personalized else "DNSMOS/sig_bak_ovr.onnx")
    ))


    rows = []
    for wav in tqdm(all_wavs, desc="DNSMOS打分中..."):
        stage_logger.log_total()
        try:
            audio, sr = sf.read(wav)
            if sr != SAMPLING_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
            fs = SAMPLING_RATE
            len_samples = int(INPUT_LENGTH * fs)
            if len(audio) < len_samples:
                audio = np.tile(audio, int(np.ceil(len_samples / len(audio))))[:len_samples]

            num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH + 1)
            hop_len_samples = fs

            sigs, baks, ovrs, p808s = [], [], [], []

            for i in range(num_hops):
                seg = audio[int(i*hop_len_samples): int((i+INPUT_LENGTH)*hop_len_samples)]
                if len(seg) < len_samples:
                    continue

                input_feat = np.array(seg, dtype=np.float32)[np.newaxis, :]
                mel_feat = np.array(audio_melspec(seg[:-160])).astype(np.float32)[np.newaxis, :, :]

                p808_mos = p808_model.run(None, {'input_1': mel_feat})[0][0][0]
                sig_raw, bak_raw, ovr_raw = main_model.run(None, {'input_1': input_feat})[0][0]

                sig, bak, ovr = get_polyfit_val(sig_raw, bak_raw, ovr_raw, personalized)
                sigs.append(sig)
                baks.append(bak)
                ovrs.append(ovr)
                p808s.append(p808_mos)

            row = {
                "filename": wav,
                "SIG": np.mean(sigs) if sigs else 0,
                "BAK": np.mean(baks) if baks else 0,
                "OVRL": np.mean(ovrs) if ovrs else 0,
                "P808_MOS": np.mean(p808s) if p808s else 0,
                "duration": len(audio)/fs,
                "valid_hops": len(sigs),
            }
            rows.append(row)
            stage_logger.log_success()
        except Exception as e:
            stage_logger.log_failed(wav, str(e))
            logger.warning(f"跳过 {wav}：{e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ 打分完成，结果保存至 {output_csv}")

    stage_logger.save()
    logger.info(f"日志已保存至 {logdir}/dnsmos_log.json")


if __name__ == "__main__":
    main()
