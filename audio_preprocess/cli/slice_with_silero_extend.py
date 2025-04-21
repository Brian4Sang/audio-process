import torch
import torchaudio
import os
import argparse
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def load_audio(path, sampling_rate=16000):
    wav, sr = torchaudio.load(str(path))
    if sr != sampling_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)(wav)
    return wav[0], sampling_rate


def save_audio(path, audio, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.cpu().numpy(), sr)


def vad_split(path, get_speech_timestamps, model, args):
    audio, sr = load_audio(path, sampling_rate=args.sampling_rate)
    total_duration = audio.shape[-1] / sr

    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sr,
        min_speech_duration_ms=int(args.min_duration * 1000),
        max_speech_duration_s=args.max_duration,
        min_silence_duration_ms=int(args.min_silence * 1000)
    )

    if not speech_timestamps:
        return []

    utt_id = Path(path).stem
    rel_path = Path(path).relative_to(args.input_dir).with_suffix('')

    output_files = []

    for idx, chunk in enumerate(speech_timestamps):
            # ✅ 第一段向前多补 0.5 秒（避免语音开头被截断）
        head_pad = 2.5 if idx == 0 else args.extend_sec
        start_sec = max(0.0, chunk['start'] / sr - head_pad)

        # start_sec = max(0.0, chunk['start'] / sr - args.extend_sec)
        end_sec = min(total_duration, chunk['end'] / sr + args.extend_sec)

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        chunk_audio = audio[start_sample:end_sample]
        out_path = args.output_dir / rel_path / f"{utt_id}_{idx:04d}.wav"
        save_audio(out_path, chunk_audio, sr)
        output_files.append(str(out_path))

    return output_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--min-silence", type=float, default=0.3)
    parser.add_argument("--extend-sec", type=float, default=0.3)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # ✅ 真正适配你环境的写法：只返回两个元素的 tuple
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    model.to(args.device)

    # ✅ 现在 utils 是个 tuple，get_speech_timestamps 是第一个元素
    get_speech_timestamps = utils[0]


    files = list(args.input_dir.rglob("*.wav"))
    print(f"🔍 Found {len(files)} wav files...")

    for file in tqdm(files, desc="VAD slicing"):
        try:
            vad_split(file, get_speech_timestamps, model, args)
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    print(f"✅ Stage 3 done. Sliced results in {args.output_dir}")


if __name__ == "__main__":
    main()
