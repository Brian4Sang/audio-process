#!/usr/bin/env python3
"""
DNSMOS 打分（支持筛选）

--input-dir       输入音频目录
--output-csv      输出评分结果 CSV
--personalized    是否启用个性化模型（pDNSMOS）
--logdir          日志输出目录
--filter          OVRL 评分阈值，筛选出 >= 此值的音频并生成 manifest

评估字段名	含义
SIG	Speech Quality：语音本身的清晰度与自然度（去噪后语音的质量）
BAK	Background Quality：背景噪声的自然度与干净程度（背景声音是否干扰）
OVRL	Overall Quality：综合主观感知得分（包括语音、背景、自然性）✅ 推荐首选
P808_MOS	模拟人类主观评分的另一路 MOS 预测（辅助参考）
"""

import os
import json
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

def score_segment(seg, p808_model, main_model, personalized):
    input_feat = np.array(seg, dtype=np.float32)[np.newaxis, :]
    mel_feat = np.array(audio_melspec(seg[:-160])).astype(np.float32)[np.newaxis, :, :]
    p808_mos = p808_model.run(None, {'input_1': mel_feat})[0][0][0]
    sig_raw, bak_raw, ovr_raw = main_model.run(None, {'input_1': input_feat})[0][0]
    sig, bak, ovr = get_polyfit_val(sig_raw, bak_raw, ovr_raw, personalized)
    return sig, bak, ovr, p808_mos

@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--output-csv", type=click.Path(), default="dnsmos_result.csv", show_default=True)
@click.option("--personalized", is_flag=True, default=False)
@click.option("--logdir", type=click.Path(), default="logs", show_default=True)
@click.option("--filter", type=float, default=None, help="筛选阈值，按 OVRL >= --filter 输出 .manifest 文件")
def main(input_dir, output_csv, personalized, logdir, filter):
    input_dir = Path(input_dir)
    all_wavs = [str(p) for p in input_dir.rglob("*.wav")]
    logger.info(f"共发现 {len(all_wavs)} 条音频")

    stage_logger = StageLogger("dnsmos", logdir)

    base_dir = Path("pretrained_models/dnsmos")
    p808_model = ort.InferenceSession(str(base_dir / "DNSMOS/model_v8.onnx"))
    main_model = ort.InferenceSession(str(
        base_dir / ("pDNSMOS/pdns_sig_bak_ovr.onnx" if personalized else "DNSMOS/sig_bak_ovr.onnx")
    ))

    rows, pass_entries, fail_entries = [], [], []

    for wav in tqdm(all_wavs, desc="DNSMOS评估中..."):
        stage_logger.log_total()
        try:
            audio, sr = sf.read(wav)
            if sr != SAMPLING_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
            fs = SAMPLING_RATE
            len_samples = int(INPUT_LENGTH * fs)

            while len(audio) < len_samples:
                audio = np.append(audio, audio)

            num_hops = int(np.floor(len(audio)/fs - INPUT_LENGTH) + 1)
            hop_len_samples = fs

            sigs, baks, ovrs, p808s = [], [], [], []
            for i in range(num_hops):
                seg = audio[int(i * hop_len_samples): int((i + INPUT_LENGTH) * hop_len_samples)]
                if len(seg) < len_samples:
                    continue
                sig, bak, ovr, p808 = score_segment(seg, p808_model, main_model, personalized)
                sigs.append(sig)
                baks.append(bak)
                ovrs.append(ovr)
                p808s.append(p808)

            row = {
                "audio_filepath": wav,
                "duration": len(audio)/fs,
                "SIG": np.mean(sigs) if sigs else 0,
                "BAK": np.mean(baks) if baks else 0,
                "OVRL": np.mean(ovrs) if ovrs else 0,
                "P808_MOS": np.mean(p808s) if p808s else 0,
                "valid_hops": len(sigs),
            }
            rows.append(row)

            # 筛选分类
            if filter is not None:
                (pass_entries if row["OVRL"] >= filter else fail_entries).append(row)

            stage_logger.log_success()
        except Exception as e:
            stage_logger.log_failed(wav, str(e))
            logger.warning(f"跳过 {wav}：{e}")

    # 写入 CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ 打分完成，结果保存至 {output_csv}")

    # 生成 manifest 路径
    outdir = Path(output_csv).parent
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "filter_pass.manifest", "w", encoding="utf-8") as f_pass, \
        open(outdir / "filter_fail.manifest", "w", encoding="utf-8") as f_fail:
        for entry in pass_entries:
            f_pass.write(json.dumps(entry, ensure_ascii=False, default=lambda o: float(o)) + "\n")
        for entry in fail_entries:
            f_fail.write(json.dumps(entry, ensure_ascii=False, default=lambda o: float(o)) + "\n")


        dur_pass = sum(e["duration"] for e in pass_entries)
        dur_fail = sum(e["duration"] for e in fail_entries)
        logger.info(f"✅ 合格音频数: {len(pass_entries)}，总时长: {dur_pass:.1f} 秒")
        logger.info(f"❌ 不合格音频数: {len(fail_entries)}，总时长: {dur_fail:.1f} 秒")

        # 所有样本的 OVRL 与 duration 提取
        ovrl_scores = [row["OVRL"] for row in rows]
        durations = [row["duration"] for row in rows]

        if ovrl_scores:
            mean_score = np.mean(ovrl_scores)
            median_score = np.median(ovrl_scores)
            dur_over_median = sum(d for o, d in zip(ovrl_scores, durations) if o > median_score)
            dur_total = sum(durations)
            ratio = dur_over_median / dur_total * 100 if dur_total > 0 else 0

            logger.info(f" 平均 OVRL 分数: {mean_score:.2f}")
            logger.info(f" 中位数 OVRL 分数: {median_score:.2f}")
            logger.info(f"📈 高于中位数的音频总时长: {dur_over_median:.1f} 秒")
            logger.info(f"📈 占全部音频总时长的比例: {ratio:.2f}%")


    stage_logger.save()
    logger.info(f"日志已保存至 {logdir}/dnsmos_log.json")

if __name__ == "__main__":
    main()
