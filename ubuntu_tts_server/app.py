# # /work-space/cookDuck_llama/ubuntu_tts_server/app.py

# import uvicorn
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# import shutil
# import os
# import logging
# import time
# import torch
# from transformers import (
#     AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline,
#     AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# )
# from melo.api import TTS

# # ==================== GPU 설정 (가장 중요) ====================
# # nvidia-smi 상에서 2번 GPU(RTX 3070)만 사용하도록 설정합니다.
# # 이 코드는 torch 라이브러리가 로드되기 전에 실행되어야 합니다.
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# # ==================== 로깅 및 기본 설정 ====================
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# # 임시 파일과 결과 파일을 저장할 디렉토리 생성
# os.makedirs("temp_files", exist_ok=True)
# os.makedirs("result_audio", exist_ok=True)

# app = FastAPI()

# # ==================== 전역 변수로 모델 및 파이프라인 저장 ====================
# # 서버가 시작될 때 이 변수들에 모델이 로드됩니다.
# stt_pipe = None
# llm_model = None
# llm_tokenizer = None
# tts_model = None
# # 이제 'cuda'는 위에서 지정한 2번 GPU를 가리키게 됩니다.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # ==================== LLM 응답 생성 함수 ====================
# def ask_llm(instruction: str, question_text: str):
#     """LLM 모델에 프롬프트를 보내고 응답을 생성하는 함수"""
#     global llm_model, llm_tokenizer
#     prompt = f"[INSTRUCTION]\n{instruction}\n[INPUT]\n{question_text}\n[OUTPUT]\n"
#     inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
#     outputs = llm_model.generate(
#         **inputs,
#         max_new_tokens=256,
#         do_sample=True,
#         top_p=0.9,
#         temperature=0.7,
#     )
#     result = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return result.split("[OUTPUT]\n")[-1].strip()

# # ==================== 서버 시작 시 모델 로딩 (핵심) ====================
# @app.on_event("startup")
# def load_models():
#     """서버가 시작될 때 모든 AI 모델을 미리 로드합니다."""
#     global stt_pipe, llm_model, llm_tokenizer, tts_model

#     logger.info("="*50)
#     logger.info("서버 시작... AI 모델을 전역 메모리에 로딩합니다.")
#     logger.info(f"사용 장치: {device} (실제 GPU: nvidia-smi 2번)", )
#     logger.info(f"데이터 타입: {torch_dtype}")
#     logger.info("="*50)
    
#     # 1. STT 모델 로딩 (Whisper)
#     logger.info("🔊 STT 모델 로딩 시작...")
#     stt_model_id = "openai/whisper-large-v3"
#     stt_model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(
#         stt_model_id, 
#         torch_dtype=torch_dtype, 
#         low_cpu_mem_usage=True, 
#         use_safetensors=True,
#         # ==================== [수정된 부분] ====================
#         # 호환성을 위해 eager attention 사용
#         attn_implementation="eager"
#         # =======================================================
#     ).to(device)
#     stt_processor = AutoProcessor.from_pretrained(stt_model_id)
#     stt_pipe = pipeline(
#         "automatic-speech-recognition",
#         model=stt_model_obj,
#         tokenizer=stt_processor.tokenizer,
#         feature_extractor=stt_processor.feature_extractor,
#         max_new_tokens=128,
#         chunk_length_s=30,
#         batch_size=16,
#         torch_dtype=torch_dtype,
#         device=device,
#     )
#     logger.info("✅ STT 모델 로딩 완료.")

#     # 2. LLM 모델 로딩 (Llama-3.2 Korean GGACHI)
#     logger.info("🧠 LLM 모델 로딩 시작...")
#     llm_model_path = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16
#     )
#     llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
#     llm_model = AutoModelForCausalLM.from_pretrained(
#         llm_model_path,
#         quantization_config=quant_config,
#         device_map="auto",
#         # ==================== [수정된 부분] ====================
#         # FlashAttention 대신 가장 안정적인 'eager' attention을 사용하도록 명시
#         attn_implementation="eager" 
#         # =======================================================
#     )
#     logger.info("✅ LLM 모델 로딩 완료.")

#     # 3. TTS 모델 로딩 (MeloTTS)
#     logger.info("🗣️ TTS 모델 로딩 시작...")
#     tts_model = TTS(language='KR', device=device)
#     logger.info("✅ TTS 모델 로딩 완료.")
    
#     logger.info("="*50)
#     logger.info("🚀 모든 모델 로딩 완료. API 서버가 요청을 받을 준비가 되었습니다.")
#     logger.info("="*50)


# # ==================== API 엔드포인트 정의 ====================
# class LLMRequest(BaseModel):
#     """LLM/TTS 요청을 위한 데이터 모델"""
#     # text: str
#     # instruction: str = "당신은 요리 레시피에 대해 친절하고 상세하게 설명해주는 AI 요리사 '쿡덕'입니다. 사용자의 질문에 맞춰 단계별로 명확하게 답변해주세요."
#     text: str
#     instruction: str = "Translate the following sentence into Korean naturally and accurately."

# @app.post("/stt")
# async def speech_to_text_endpoint(audio: UploadFile = File(...)):
#     """오디오 파일을 받아 텍스트로 변환하여 JSON으로 반환"""
#     if not stt_pipe:
#         raise HTTPException(status_code=503, detail="STT 모델이 아직 로딩되지 않았습니다.")

#     temp_input_path = f"temp_files/{audio.filename}"
#     try:
#         with open(temp_input_path, "wb") as buffer:
#             shutil.copyfileobj(audio.file, buffer)
        
#         logger.info(f"STT 처리 시작: {temp_input_path}")
#         stt_result = stt_pipe(temp_input_path, generate_kwargs={"language": "korean"})
#         recognized_text = stt_result["text"]
#         logger.info(f"STT 처리 완료: {recognized_text}")

#         return {"text": recognized_text}
#     except Exception as e:
#         logger.error(f"STT 처리 중 에러 발생: {e}")
#         raise HTTPException(status_code=500, detail=f"STT 처리 중 에러 발생: {e}")
#     finally:
#         if os.path.exists(temp_input_path):
#             os.remove(temp_input_path)

# @app.post("/generate-speech")
# async def generate_speech_endpoint(request: LLMRequest):
#     """텍스트 프롬프트를 받아 LLM 답변을 생성하고 TTS로 변환하여 음성 파일을 반환"""
#     if not llm_model or not tts_model:
#         raise HTTPException(status_code=503, detail="LLM 또는 TTS 모델이 아직 로딩되지 않았습니다.")

#     try:
#         logger.info(f"LLM 처리 시작. Instruction: {request.instruction}, Text: {request.text}")
#         response_text = ask_llm(request.instruction, request.text)
#         logger.info(f"LLM 처리 완료: {response_text}")

#         logger.info(f"TTS 처리 시작...")
#         output_filename = f"response_{int(time.time())}.wav"
#         output_path = f"result_audio/{output_filename}"
        
#         speaker_ids = tts_model.hps.data.spk2id
#         tts_model.tts_to_file(response_text, speaker_ids['KR'], output_path, speed=1.1)
#         logger.info(f"TTS 처리 완료: {output_path}")

#         return FileResponse(path=output_path, media_type="audio/wav", filename=output_filename)
#     except Exception as e:
#         logger.error(f"LLM/TTS 처리 중 에러 발생: {e}")
#         raise HTTPException(status_code=500, detail=f"LLM/TTS 처리 중 에러 발생: {e}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
# /work-space/cookDuck_llama/ubuntu_tts_server/app.py

# /work-space/cookDuck_llama/ubuntu_tts_server/app.py

# /work-space/cookDuck_llama/ubuntu_tts_server/app.py

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import shutil
import os
import logging
import time
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline,
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from melo.api import TTS
# from detect import IngredientsDetect

# --- GPU, 로깅, 기본 설정 (이전과 동일) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("temp_files", exist_ok=True)
os.makedirs("result_audio", exist_ok=True)
app = FastAPI()

# --- 전역 모델 변수 (이전과 동일) ---
stt_pipe, llm_model, llm_tokenizer, tts_model = None, None, None, None
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # yolo 모델 로딩
# yolo_model = IngredientsDetect() 

# def predect_image():
#     result = yolo_model(20)

#     return result

# --- LLM 응답 생성 함수 (이전과 동일) ---
def ask_llm(instruction: str, question_text: str):
    global llm_model, llm_tokenizer
    prompt = f"[INSTRUCTION]\n{instruction}\n[INPUT]\n{question_text}\n[OUTPUT]\n"
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=0.7)
    result = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("[OUTPUT]\n")[-1].strip()

# --- 서버 시작 시 모델 로딩 (이전과 동일) ---
@app.on_event("startup")
def load_models():
    # ... (이전과 동일한 모델 로딩 로직)
    global stt_pipe, llm_model, llm_tokenizer, tts_model
    logger.info("="*50); logger.info("서버 시작... AI 모델을 전역 메모리에 로딩합니다."); logger.info(f"사용 장치: {device} (실제 GPU: nvidia-smi 2번)"); logger.info(f"데이터 타입: {torch_dtype}"); logger.info("="*50)
    logger.info("🔊 STT 모델 로딩 시작..."); stt_model_id = "openai/whisper-large-v3"; stt_model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(stt_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="eager").to(device); stt_processor = AutoProcessor.from_pretrained(stt_model_id); stt_pipe = pipeline("automatic-speech-recognition", model=stt_model_obj, tokenizer=stt_processor.tokenizer, feature_extractor=stt_processor.feature_extractor, max_new_tokens=128, chunk_length_s=30, batch_size=16, torch_dtype=torch_dtype, device=device); logger.info("✅ STT 모델 로딩 완료.")
    logger.info("🧠 LLM 모델 로딩 시작..."); llm_model_path = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"; quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16); llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path); llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, quantization_config=quant_config, device_map="auto", attn_implementation="eager"); logger.info("✅ LLM 모델 로딩 완료.")
    logger.info("🗣️ TTS 모델 로딩 시작..."); tts_model = TTS(language='KR', device=device); logger.info("✅ TTS 모델 로딩 완료.")
    logger.info("="*50); logger.info("🚀 모든 모델 로딩 완료. API 서버가 요청을 받을 준비가 되었습니다."); logger.info("="*50)


# ==================== API 엔드포인트 정의 (수정됨) ====================
class LLMRequest(BaseModel):
    text: str
    instruction: str = "I'm an AI assistant, which means I'm a computer program designed to simulate conversation and answer questions to the best of my ability. I'm here to help you with any questions or tasks you may have, and I'll do my best to provide you with accurate and helpful information."

    # text: str
    # instruction: str = "Translate the following sentence into Korean naturally and accurately."
    
# STT 엔드포인트 (이전과 동일)
@app.post("/stt")
async def speech_to_text_endpoint(audio: UploadFile = File(...)):
    # ... (이전과 동일한 STT 로직)
    if not stt_pipe: raise HTTPException(status_code=503, detail="STT 모델이 아직 로딩되지 않았습니다.")
    temp_input_path = f"temp_files/{audio.filename}";  
    try:
        with open(temp_input_path, "wb") as buffer: shutil.copyfileobj(audio.file, buffer)
        logger.info(f"STT 처리 시작: {temp_input_path}"); stt_result = stt_pipe(temp_input_path, generate_kwargs={"language": "korean"}); recognized_text = stt_result["text"]; logger.info(f"STT 처리 완료: {recognized_text}")
        return {"text": recognized_text}
    except Exception as e: logger.error(f"STT 처리 중 에러 발생: {e}"); raise HTTPException(status_code=500, detail=f"STT 처리 중 에러 발생: {e}")
    finally:
        if os.path.exists(temp_input_path): os.remove(temp_input_path)


# LLM 응답 생성 엔드포인트
@app.post("/generate-llm-response")
async def generate_llm_response_endpoint(request: LLMRequest):
    """LLM 답변 텍스트를 생성하고, 생성된 텍스트와 오디오 파일명을 반환"""
    if not llm_model or not tts_model:
        raise HTTPException(status_code=503, detail="LLM 또는 TTS 모델이 아직 로딩되지 않았습니다.")
    try:
        logger.info(f"LLM 처리 시작. Instruction: {request.instruction}, Text: {request.text}")
        response_text = ask_llm(request.instruction, request.text)
        logger.info(f"LLM 처리 완료: {response_text}")

        logger.info(f"TTS 파일 생성 시작...")
        output_filename = f"response_{int(time.time())}.wav"
        output_path = f"result_audio/{output_filename}"
        
        speaker_ids = tts_model.hps.data.spk2id
        tts_model.tts_to_file(response_text, speaker_ids['KR'], output_path, speed=1.1)
        logger.info(f"TTS 파일 생성 완료: {output_path}")

        return JSONResponse(content={"llm_text": response_text, "audio_filename": output_filename})
    except Exception as e:
        logger.error(f"LLM/TTS 처리 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"LLM/TTS 처리 중 에러 발생: {e}")

# [신규] 초기 인사말 생성 엔드포인트
@app.post("/generate-greeting")
async def generate_greeting_endpoint():
    """앱 시작 시 초기 인사말 음성을 생성하고 텍스트와 파일명을 반환"""
    if not tts_model:
        raise HTTPException(status_code=503, detail="TTS 모델이 아직 로딩되지 않았습니다.")
    
    try:
        greeting_text = "안녕하세요! AI 요리사 쿡덕입니다. 어떤 요리의 레시피를 알려드릴까요?"
        logger.info(f"초기 인사말 생성: {greeting_text}")

        output_filename = f"greeting_{int(time.time())}.wav"
        output_path = f"result_audio/{output_filename}"
        
        speaker_ids = tts_model.hps.data.spk2id
        tts_model.tts_to_file(greeting_text, speaker_ids['KR'], output_path, speed=1.1)
        logger.info(f"인사말 TTS 파일 생성 완료: {output_path}")

        return JSONResponse(content={"llm_text": greeting_text, "audio_filename": output_filename})
    except Exception as e:
        logger.error(f"인사말 생성 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"인사말 생성 중 에러 발생: {e}")


# 오디오 파일 제공 엔드포인트
@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """`result_audio` 디렉토리에서 오디오 파일을 찾아 반환"""
    file_path = f"result_audio/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
    return FileResponse(path=file_path, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)


