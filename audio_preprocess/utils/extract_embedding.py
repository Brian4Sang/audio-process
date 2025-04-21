import onnxruntime
import numpy as np
import soundfile as sf
import librosa

# utils/extract_embedding.py 中改造 extract_embedding
def extract_embedding(wav_path, model_path, chunk_duration=2.0, sr=16000):
    import librosa
    import onnxruntime
    import numpy as np
    import soundfile as sf

    # 读取并转为单通道
    audio, orig_sr = sf.read(wav_path)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    chunk_size = int(chunk_duration * sr)
    total_len = len(audio)
    embeddings = []

    sess = onnxruntime.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # 每段提取 embedding
    for i in range(0, total_len, chunk_size):
        chunk = audio[i: i + chunk_size]
        if len(chunk) < sr:  # 太短跳过
            continue

        # 特征提取
        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=sr,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=80,
            power=2,
        )
        log_mel = librosa.power_to_db(mel).T.astype(np.float32)  # [T, 80]
        log_mel = np.expand_dims(log_mel, axis=0)  # [1, T, 80]

        emb = sess.run([output_name], {input_name: log_mel})[0]  # [1, D]
        embeddings.append(emb.squeeze())

    return np.stack(embeddings) if embeddings else np.empty((0, 192))
