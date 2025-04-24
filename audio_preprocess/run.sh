#!/bin/bash
set -e

stage=7
stop_stage=7

# 原始音频路径（支持 mp3/mp4/wav）
raw_input_dir="/workspace/workdir/tts_data_traning/LibriTTS/ximalaya/mp3test"

# 项目路径
project_dir="/brian_f/audio-pipeline"

# 统一的数据输出根目录/${project_dir}/data_root
data_root="data"

# Stage 0 输出目录（转为 wav）
wav_dir="${data_root}/to_wav"

# Stage 1 输出目录（重采样）
sampling_rate=44100
resample_tag=$((sampling_rate / 1000))k
resample_dir="${data_root}/resample_${resample_tag}"

# Stage 2 输出目录（人声分离）
separate_dir="${data_root}/separate_vocals_${resample_tag}"

# Stage 3 输出目录（切片）
slice_dir="${data_root}/sliced_silero"

# 日志目录
log_root="logs"

embedding_model="/brian_f/audio-pipeline/audio_preprocess/pretrained_models/campplus.onnx"

# Stage 0: Convert to wav (no resample)
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Converting input files to wav (no resample)...\
        you will get .wav in data/to_wav/"

  python -m audio_preprocess.cli.convert_to_wav \
    "${raw_input_dir}" "${wav_dir}" \
    --recursive \
    --segment 0 \
    --logdir "${log_root}"

  echo "Stage 0 done. Outputs saved to: ${wav_dir}"
fi

# Stage 1: Resample to 自定义的采样率（如 16000、24000、44100）声道数(mono->1、nomo->2)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo " Stage 1: Resample audio to 24kHz, mono...\
          you will get resampled ${sampling_rate} .wav in ${resample_dir}"

  python -m audio_preprocess.cli.resample \
    "${wav_dir}" "${resample_dir}" \
    --recursive \
    --sampling-rate ${sampling_rate} \
    --nomo \
    --overwrite \
    --num-workers 8 \
    --subtype PCM_16 \
    --log-dir "${log_root}"

  echo "Stage 1 done. Outputs saved to: ${resample_dir}"
fi

# Stage 2: Separate vocals using Demucs

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Separate vocals from mixture...\
  you will get separated vocals in data/eparate_vocals_${resample_tag}"

  python -m audio_preprocess.cli.separate_audio \
    "${resample_dir}" "${separate_dir}" \
    --track vocals \
    --model htdemucs \
    --overwrite \
    --recursive \
    --num_workers_per_gpu 2 \
    --logdir "${log_root}" \
    --strict-check

  echo "Stage 2 done. Outputs saved to: ${separate_dir}"
fi

# # Stage 3: Slice into short utterances (3~12s)
# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#   echo "Stage 3: Slicing long audio into short chunks ...\
#         you will get sliced piece in ${slice_dir}"
#   python -m audio_preprocess.cli.slice_audio \
#     "${separate_dir}" "${slice_dir}" \
#     --min-duration 3.0 \
#     --max-duration 10.0 \
#     --min-silence-duration 0.3 \
#     --max-silence-kept 0.5 \
#     --top-db -25 \
#     --hop-length 10 \
#     --merge-short \
#     --overwrite \
#     --logdir "${log_root}"

#   echo "Stage 3 done. Sliced audio saved to: ${slice_dir}"
# fi

# Stage 3: Slice with Silero VAD + 0.3s extension
slice_mode="silero"  # 可选: silero / energy / vad_extend 等

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo -e " Stage 3: Audio slicing using mode=${slice_mode} ... 
            you will get sliced piece in ${slice_dir}"

  if [ "$slice_mode" == "silero" ]; then
    python cli/slice_with_silero_extend.py \
      "${separate_dir}" "${slice_dir}" \
      --min-duration 2.0 \
      --max-duration 15.0 \
      --min-silence 0.3 \
      --extend-sec 0.3 \
      --device cpu
  elif [ "$slice_mode" == "energy" ]; then
    python -m audio_preprocess.cli.slice_audio \
      "${separate_dir}" "${slice_dir}" \
      --min-duration 3.0 \
      --max-duration 10.0 \
      --min-silence-duration 0.3 \
      --max-silence-kept 0.5 \
      --top-db -25 \
      --hop-length 10 \
      --merge-short \
      --overwrite \
      --logdir "${log_root}"
  else
    echo "Unsupported slice_mode: $slice_mode"
    exit 1
  fi

  echo " Stage 3 done. Sliced results in ${slice_dir}"
fi



# Stage 4: Multi-spk detection
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Multi-speaker detection... \
        you will get resluts in ${log_root}/multi_spk_pred.json"

  python -m audio_preprocess.cli.multi_spk_detect \
    --input-dir ${slice_dir} \
    --embedding-model ${embedding_model} \
    --output-json ${log_root}/multi_spk_pred.json \
    --logdir logs

  echo " Stage 4 done."
fi

# Stage 5: Loudness Normalization
  # sliced_dir="${data_root}/sliced"
  volnorm_dir="${data_root}/vol_norm"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo -e " Stage 5: Loudness normalization (LUFS + Peak)...\n
          you will get norm-volume wav in ${volnorm_dir}"

  # 设置响度参数（可调节）
  target_loudness=-23.0   # LUFS
  peak_level=-1.0         # dBFS
  block_size=0.4          # 每块用于评估 loudness 的窗口大小（秒）

 

  python -m audio_preprocess.cli.loudness_norm \
    "${slice_dir}" "${volnorm_dir}" \
    --recursive \
    --overwrite \
    --num-workers 8 \
    --loudness ${target_loudness} \
    --peak ${peak_level} \
    --block-size ${block_size} \
    --logdir "${log_root}"

  echo " Stage 5 done. Normalized audio saved to: ${volnorm_dir}"
fi

# Stage 6: Transcribe audio to text with FunASR
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo -e " Stage 6: Transcribe audio using FunASR with punctuation
            you will get .lab in the same dir"

  python -m audio_preprocess.cli.transcribe \
    ${volnorm_dir} \
    --recursive \
    --lang zh \
    --model-type funasr \
    --model-size paraformer-zh \
    --num-workers 1

  echo " Stage 6 done. Transcriptions saved as .lab files in the audio path"
fi

# Stage 7: Cluster speakers based on embedding
  cluster_dir="${data_root}/cluster_spk"\
  method="dbscan"
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo -e " Stage 7: Cluster speakers using ECAPA embedding + Agglomerative clustering
            you will get utt2spk.json and audio organized by speakers in ${cluster_dir}"

  python -m audio_preprocess.cli.multi_spk_cluster \
    --input-dir ${volnorm_dir}\
    --embedding-model ${embedding_model} \
    --output-json ${cluster_dir}/utt2spk.json \
    --output-split-dir ${cluster_dir} \
    --logdir logs \
    --method ${method} \
    --recursive

  echo " Stage 7 done. Speaker clusters written to utt2spk.json, audios grouped under data/split_by_spk"
fi

# Stage 8: Compute DNSMOS scores
dnsmos_csv="results/dnsmos_scores.csv"
personalized_flag=""
# 个性化模型
# personalized_flag="--personalized"

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo -e " Stage 8: Compute DNSMOS scores for audio quality evaluation
            Output: CSV=${dnsmos_csv}, logs in ${dnsmos_logdir}"

  python -m audio_preprocess.cli.dnsmos_score \
    --input-dir ${volnorm_dir} \
    --output-csv ${dnsmos_csv} \
    --logdir logs \
    ${personalized_flag}

  echo " Stage 8 done. DNSMOS scores written to ${dnsmos_csv}"
fi




