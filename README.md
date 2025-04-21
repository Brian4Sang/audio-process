# audio-process
## To get high-quality tts training dataset from raw audio/video

## 项目结构
audio-pipeline/
├── run.sh                         # 主处理脚本（多阶段控制）
├── data/                          # 所有处理结果将保存在此目录
├── logs/                          # 各阶段日志保存目录
└── fish_audio_preprocess/cli/    # 各阶段 CLI 接口（需安装依赖）

### 安装项目依赖
pip install -e fish_audio_preprocess
...
### 各阶段说明
0	Convert to WAV	将原始 mp3/mp4/wav 转为标准 WAV 格式
1	Resample	重采样至目标采样率（默认 44100 Hz 用于stage 2 分离人声）
2	Vocal Separation	使用 Demucs 分离人声
3	Slice	使用 Silero VAD 切分音频片段
4	Multi-speaker Detection	使用 Resemblyzer/CAMPPlus 检测多说话人
5	Loudness Normalization	LUFS + Peak 音量归一化处理
6	ASR Transcription	使用 FunASR 进行语音识别转写文本
...
待做【切片效果：完整句子和字、文本识别：标点优化】【多说话人识别、数据质量分类】
### 参考
1. https://github.com/fishaudio/audio-preprocess
2. https://wenetspeech4tts.github.io/wenetspeech4tts/
3. https://github.com/FireRedTeam/FireRedTTS
