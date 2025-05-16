# import re
# from pathlib import Path
# from typing import Literal

# from loguru import logger
# from tqdm import tqdm

# PROMPT = {
#     "zh": "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。",
#     "en": "In the realm of advanced technology, the evolution of artificial intelligence stands as a monumental achievement."
# }

# ASRModelType = Literal["paraformer", "sensevoice"]

# #
# def batch_transcribe(
#     files: list[Path],
#     # model_size: str,
#     model_type: ASRModelType,
#     lang: str,
#     pos: int,
#     compute_type: str,
#     batch_size: int = 1,
# ):
#     results = {}
#     # if model_type == "whisper":
#     #     from faster_whisper import WhisperModel

#     #     if lang == "jp":
#     #         lang = "ja"
#     #         logger.info(
#     #             f"Language {lang} is not supported by whisper, using ja(japenese) instead"
#     #         )

#     #     logger.info(f"Loading {model_size} model for {lang} transcription")
#     #     kwargs = {}
#     #     if not batch_size or batch_size == 1:
#     #         model = WhisperModel(model_size, compute_type=compute_type)
#     #     else:
#     #         from faster_whisper.transcribe import BatchedInferencePipeline

#     #         model = BatchedInferencePipeline(model_size, compute_type=compute_type)
#     #         kwargs["batch_size"] = batch_size
#     #     for file in tqdm(files, position=pos):
#     #         if lang in PROMPT:
#     #             result = model.transcribe(
#     #                 file, language=lang, initial_prompt=PROMPT[lang], **kwargs
#     #             )
#     #         else:
#     #             result = model.transcribe(file, language=lang, **kwargs)
#     #         result = list(result)
#     #         results[str(file)] = result["text"]
# # 
#     if model_type == "paraformer":
#         from funasr import AutoModel
#         from funasr.utils.postprocess_utils import rich_transcription_postprocess

#         logger.info(f"Loading paraformer model for {lang} transcription")
#         model = AutoModel(
#             model="paraformer-zh" if lang =="zh" else "paraformer-en",
#             vad_model="fsmn-vad",
#             # punc_model="ct-punc-c" if lang =="zh" else None,
#             log_level="ERROR",
#             disable_pbar=True,
#         )
#         for file in tqdm(files, position=pos):
#             if lang in PROMPT:
#                 result = model.generate(
#                     input=file,
#                     batch_size_s=300,
#                     hotword=PROMPT[lang],
#                     merge_vad=True,
#                     merge_length_s=15,
#                 )
#             else:
#                 result = model.generate(input=file, batch_size_s=300)
#             # print(result)
#             if isinstance(result, list):
#                 results[str(file)] = "".join(
#                     [re.sub(r"<\|.*?\|>", "", item["text"]) for item in result]
#                 )
#             else:
#                 results[str(file)] = result["text"]

#     elif model_type == "sensevoice":
#         from funasr import AutoModel
#         from funasr.utils.postprocess_utils import rich_transcription_postprocess

#         model_dir = "iic/SenseVoiceSmall"

#         logger.info(f"Loading {model_type} model for {lang} transcription")
#         model = AutoModel(
#             model=model_dir,
#             trust_remote_code=True,
#             remote_code="./model.py",
#             vad_model="fsmn-vad",
#             vad_kwargs={"max_single_segment_time": 30000},
#             # punc_model="ct-punc" if model_size == "paraformer-zh" else None,
#             log_level="ERROR",
#             device="cuda:0",
#         )

#         for file in tqdm(files, position=pos):
#             if lang in PROMPT:
#                 result = model.generate(
#                     input=file,
#                     cache={},
#                     batch_size_s=60,
#                     language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
#                     use_itn=True,
#                     hotword=PROMPT[lang],
#                     merge_vad=True,
#                     merge_length_s=15,
#                 )
#             else:
#                 result = model.generate(input=file, batch_size_s=300)
#             # print(result)
#             if isinstance(result, list):
#                 results[str(file)] = "".join(
#                     [re.sub(r"<\|.*?\|>", "", item["text"]) for item in result]
#                 )
#             else:
#                 results[str(file)] = result["text"]
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")
#     return results

# # import re
# # import json
# # from pathlib import Path
# # from typing import Literal

# # from loguru import logger
# # from tqdm import tqdm

# # PROMPT = {
# #     "zh": "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。",
# #     "en": "In the realm of advanced technology, the evolution of artificial intelligence stands as a monumental achievement."
# # }

# # ASRModelType = Literal["funasr", "sensevoice"]

# # def batch_transcribe(
# #     files: list[Path],
# #     model_size: str,
# #     lang: str,
# #     pos: int,
# #     compute_type: str,
# #     batch_size: int = 1,
# #     manifest_path: Path = Path("asr_output.manifest"),  # manifest 路径参数
# # ):
# #     results = {}

# #     # 打开 manifest 文件并准备写入
# #     with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
# #         for file in tqdm(files, position=pos):
# #             file_result = {"audio_filepath": str(file), "paraformer": "", "sensevoice": ""}
            
# #             # 使用 Paraformer 模型进行转录
# #             from funasr import AutoModel
# #             from funasr.utils.postprocess_utils import rich_transcription_postprocess

# #             logger.info(f"Loading paraformer model for {lang} transcription")

# #             model = AutoModel(
# #                 model="paraformer-zh" if lang =="zh" else "paraformer-en"
# #                 vad_model="fsmn-vad",
# #                 # punc_model="ct-punc-c" if lang =="zh" else None,
# #                 log_level="ERROR",
# #                 disable_pbar=True,
# #             )
# #             if lang in PROMPT:
# #                 result = model.generate(
# #                     input=file,
# #                     batch_size_s=300,
# #                     hotword=PROMPT[lang],
# #                     merge_vad=True,
# #                     merge_length_s=15,
# #                 )
# #             else:
# #                 result = model.generate(input=file, batch_size_s=300)
            
# #             if isinstance(result, list):
# #                 file_result["paraformer"] = "".join(
# #                     [re.sub(r"<\|.*?\|>", "", item["text"]) for item in result]
# #                 )
# #             else:
# #                 file_result["paraformer"] = result["text"]

# #             # 对 SenseVoice 模型进行转录
# #             model_dir = "iic/SenseVoiceSmall"
# #             logger.info(f"Loading SenseVoice model for  {lang} transcription")
# #             model = AutoModel(
# #                 model=model_dir,
# #                 trust_remote_code=True,
# #                 remote_code="./model.py",
# #                 vad_model="fsmn-vad",
# #                 vad_kwargs={"max_single_segment_time": 30000},
# #                 punc_model="ct-punc" if lang == "zh" else None,
# #                 log_level="ERROR",
# #                 device="cuda:0",
# #             )

# #             if lang in PROMPT:
# #                 result = model.generate(
# #                     input=file,
# #                     cache={},
# #                     batch_size_s=60,
# #                     language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
# #                     use_itn=True,
# #                     hotword=PROMPT[lang],
# #                     merge_vad=True,
# #                     merge_length_s=15,
# #                 )
# #             else:
# #                 result = model.generate(input=file, batch_size_s=300)
            
# #             if isinstance(result, list):
# #                 file_result["sensevoice"] = "".join(
# #                     [re.sub(r"<\|.*?\|>", "", item["text"]) for item in result]
# #                 )
# #             else:
# #                 file_result["sensevoice"] = result["text"]

# #             # 将每个音频文件的结果记录到 manifest 文件
# #             manifest_file.write(json.dumps(file_result, ensure_ascii=False) + "\n")

# #             # 记录到 results 字典
# #             results[str(file)] = file_result

# #     return results


import re
from pathlib import Path
from typing import Literal

from loguru import logger
from tqdm import tqdm

PROMPT = {
    "zh": "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。",
    "en": "In the realm of advanced technology, the evolution of artificial intelligence stands as a monumental achievement.",
    "ja": "先進技術の領域において、人工知能の進化は画期的な成果として立っています。常に機械ができることの限界を押し広げているこのダイナミックな分野は、急速な成長と革新を見せています。複雑なデータパターンの解読から自動運転車の操縦まで、AIの応用は広範囲に及びます。",
}

ASRModelType = Literal["funasr", "whisper"]


def batch_transcribe(
    files: list[Path],
    # model_size: str,
    model_type: ASRModelType,
    lang: str,
    pos: int,
    compute_type: str,
    batch_size: int = 1,
):
    results = {}
    if model_type == "whisper":
        from faster_whisper import WhisperModel

        if lang == "jp":
            lang = "ja"
            logger.info(
                f"Language {lang} is not supported by whisper, using ja(japenese) instead"
            )

        logger.info(f"Loading  model for {lang} transcription")
        kwargs = {}
        if not batch_size or batch_size == 1:
            model = WhisperModel(model_size, compute_type=compute_type)
        else:
            from faster_whisper.transcribe import BatchedInferencePipeline

            model = BatchedInferencePipeline(model_size, compute_type=compute_type)
            kwargs["batch_size"] = batch_size
        for file in tqdm(files, position=pos):
            if lang in PROMPT:
                result = model.transcribe(
                    file, language=lang, initial_prompt=PROMPT[lang], **kwargs
                )
            else:
                result = model.transcribe(file, language=lang, **kwargs)
            result = list(result)
            results[str(file)] = result["text"]

    elif model_type == "funasr":
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        logger.info(f"Loading {model_size} model for {lang} transcription")
        model = AutoModel(
            model=model_size,
            vad_model="fsmn-vad",
            punc_model="ct-punc" if model_size == "paraformer-zh" else None,
            log_level="ERROR",
            disable_pbar=True,
        )
        for file in tqdm(files, position=pos):
            if lang in PROMPT:
                result = model.generate(
                    input=file,
                    batch_size_s=300,
                    hotword=PROMPT[lang],
                    merge_vad=True,
                    merge_length_s=15,
                )
            else:
                result = model.generate(input=file, batch_size_s=300)
            # print(result)
            if isinstance(result, list):
                results[str(file)] = "".join(
                    [re.sub(r"<\|.*?\|>", "", item["text"]) for item in result]
                )
            else:
                results[str(file)] = result["text"]

    elif model_type == "sensevoice":
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        model_dir = "iic/SenseVoiceSmall"

        logger.info(f"Loading {model_type} model for {lang} transcription")
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            log_level="ERROR",
            device="cuda:0",
        )

        for file in tqdm(files, position=pos):
            if lang in PROMPT:
                result = model.generate(
                    input=file,
                    cache={},
                    batch_size_s=60,
                    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=True,
                    hotword=PROMPT[lang],
                    merge_vad=True,
                    merge_length_s=15,
                )
            else:
                result = model.generate(input=file, batch_size_s=300)
            # print(result)
            if isinstance(result, list):
                results[str(file)] = "".join(
                    [re.sub(r"<\|.*?\|>", "", item["text"]) for item in result]
                )
            else:
                results[str(file)] = result["text"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return results