# /work-space/cookDuck_llama/ubuntu_tts_server/prompt_app.py
# 이 파일은 Llama 3.2의 공식 프롬프트 형식을 테스트하기 위한 새로운 버전입니다.

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

# --- 기본 설정 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RESULT_AUDIO_DIR = "result_audio"
os.makedirs("temp_files", exist_ok=True)
os.makedirs(RESULT_AUDIO_DIR, exist_ok=True)
app = FastAPI()

# --- 모델 및 함수 정의 ---
stt_pipe, llm_model, llm_tokenizer, tts_model = None, None, None, None
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def ask_llm(instruction: str, question_text: str):
    """Llama 3.2의 공식 형식에 맞춰 프롬프트를 구성하고 모델에 전달하는 함수"""
    global llm_model, llm_tokenizer

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question_text},
    ]

    prompt = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    logger.info(f"Llama 3.2 최종 프롬프트:\n{prompt}")

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.7)
    
    response_text = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response_text.strip()

@app.on_event("startup")
def load_models():
    """서버 시작 시 모든 AI 모델을 미리 로드합니다."""
    global stt_pipe, llm_model, llm_tokenizer, tts_model
    logger.info("모델 로딩 시작...");
    stt_model_id = "openai/whisper-large-v3"; stt_model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(stt_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="eager").to(device); stt_processor = AutoProcessor.from_pretrained(stt_model_id); stt_pipe = pipeline("automatic-speech-recognition", model=stt_model_obj, tokenizer=stt_processor.tokenizer, feature_extractor=stt_processor.feature_extractor, max_new_tokens=128, chunk_length_s=30, batch_size=16, torch_dtype=torch_dtype, device=device)
    llm_model_path = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"; quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16); llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path); llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, quantization_config=quant_config, device_map="auto", attn_implementation="eager")
    tts_model = TTS(language='KR', device=device)
    logger.info("🚀 모든 모델 로딩 완료.")

# ==================== [핵심 수정 부분] ====================
class LLMRequest(BaseModel):
    text: str
    # 'instruction'을 다양한 질문 유형에 대응하도록 훨씬 더 구체적이고 구조적으로 수정합니다.
    instruction: str = (
        "You are 'CookDuck', an expert and friendly AI cooking assistant. Your answers must always be in Korean.\n"
        "Your primary role is to provide accurate, concise, and helpful information for any cooking-related query. You must strictly follow these rules:\n\n"
        "1.  **Handle Different Question Types:**\n"
        "    - **For 'How-To' Questions (e.g., 'How long to cook?'):** Give a direct, step-by-step, or factual answer. Provide clear indicators like time, temperature, or visual cues.\n"
        "    - **For 'Recommendation' Questions (e.g., 'Recommend hamburgers'):** Suggest 3-5 distinct and popular examples. For each, provide a brief, one-sentence description. Do not just list names.\n"
        "    - **For 'General Knowledge' Questions (e.g., 'What is sous vide?'):** Provide a clear and simple definition.\n\n"
        "2.  **Maintain High Quality:**\n"
        "    - **NO REPETITION:** Never repeat the same phrase or sentence multiple times. Provide diverse and informative content.\n"
        "    - **BE FACTUAL:** Base your answers on general culinary knowledge. If you do not know the answer, you MUST state '죄송하지만, 그 정보는 제가 가지고 있지 않아요.' (Sorry, I don't have that information).\n"
        "    - **BE CONCISE:** Get straight to the point. Avoid unnecessary introductory phrases. Do not repeat the user's question.\n\n"
        "3.  **Tone & Language:**\n"
        "    - Maintain a friendly and encouraging tone.\n"
        "    - Your entire response MUST be in Korean."
    )
# =============================================================================

# --- API 엔드포인트 ---

@app.post("/stt")
async def speech_to_text_endpoint(audio: UploadFile = File(...)):
    if not stt_pipe: raise HTTPException(status_code=503, detail="STT 모델이 아직 로딩되지 않았습니다.")
    temp_input_path = f"temp_files/{audio.filename}"; 
    try:
        with open(temp_input_path, "wb") as buffer: shutil.copyfileobj(audio.file, buffer)
        stt_result = stt_pipe(temp_input_path, generate_kwargs={"language": "korean"}); return {"text": stt_result["text"]}
    finally:
        if os.path.exists(temp_input_path): os.remove(temp_input_path)


@app.post("/generate-llm-response")
async def generate_llm_response_endpoint(request: LLMRequest):
    if not llm_model or not tts_model: raise HTTPException(status_code=503, detail="LLM 또는 TTS 모델이 아직 로딩되지 않았습니다.")
    response_text = ask_llm(request.instruction, request.text)
    output_filename = f"response_{int(time.time())}.wav"
    output_path = os.path.join(RESULT_AUDIO_DIR, output_filename)
    speaker_ids = tts_model.hps.data.spk2id
    tts_model.tts_to_file(response_text, speaker_ids['KR'], output_path, speed=1.1)
    return JSONResponse(content={"llm_text": response_text, "audio_filename": output_filename})

@app.post("/generate-greeting")
async def generate_greeting_endpoint():
    if not tts_model: raise HTTPException(status_code=503, detail="TTS 모델이 아직 로딩되지 않았습니다.")
    greeting_text = "안녕하세요! AI 요리사 쿡덕입니다." # 인사말도 Q&A에 맞게 수정
    output_filename = "greeting.wav" 
    output_path = os.path.join(RESULT_AUDIO_DIR, output_filename)
    if not os.path.exists(output_path):
        speaker_ids = tts_model.hps.data.spk2id
        tts_model.tts_to_file(greeting_text, speaker_ids['KR'], output_path, speed=1.)
    return JSONResponse(content={"llm_text": greeting_text, "audio_filename": output_filename})

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    file_path = os.path.join(RESULT_AUDIO_DIR, filename)
    if not os.path.exists(file_path): raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
    return FileResponse(path=file_path, media_type="audio/wav")

@app.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    if filename == "greeting.wav":
        return JSONResponse(content={"message": "Greeting file is not deleted."}, status_code=200)
    try:
        file_path = os.path.join(RESULT_AUDIO_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return JSONResponse(content={"message": f"File {filename} deleted successfully."}, status_code=200)
        else:
            return JSONResponse(content={"detail": "File not found"}, status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
