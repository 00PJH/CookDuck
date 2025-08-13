import os
import time
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline,
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from melo.api import TTS

# ==================== GPU 설정 ====================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # RTX 3070을 사용하려면 2로 설정

# ==================== GPU 사용량 체크 함수 ====================
def get_gpu_memory():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**2)
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024**2)
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024**2)
        free = reserved - allocated
        name = torch.cuda.get_device_name(gpu_id)
        return {
            "GPU ID": gpu_id,
            "GPU Name": name,
            "Total MB": round(total, 2),
            "Reserved MB": round(reserved, 2),
            "Allocated MB": round(allocated, 2),
            "Free MB": round(free, 2),
        }
    return {"GPU": "not available"}

# ==================== LLM 응답 함수 정의 ====================
def ask_cooking_question(instruction, question_text, model, tokenizer):
    prompt = f"[INSTRUCTION]\n{instruction}\n[INPUT]\n{question_text}\n[OUTPUT]\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("[OUTPUT]\n")[-1].strip()

# ==================== 전체 처리 시작 ====================
total_start = time.perf_counter()

# ===== [1] STT =====
print("\n🔊 STT 시작")
stt_model_id = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    stt_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
stt_processor = AutoProcessor.from_pretrained(stt_model_id)
stt_pipe = pipeline(
    "automatic-speech-recognition",
    model=stt_model,
    tokenizer=stt_processor.tokenizer,
    feature_extractor=stt_processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# 영어 번역 질문 테스트
stt_start = time.perf_counter() # text넣기 시작
stt_result = stt_pipe("./stt-llm-pipeline/audio/output_en_test.wav", generate_kwargs={"language": "korean"})
stt_text = stt_result["text"]
stt_end = time.perf_counter()
print(f"📝 STT 결과: {stt_text}")
print(f"⏱️ STT 시간: {stt_end - stt_start:.2f}초")

# ===== [2] LLM =====
print("\n🧠 LLM 시작")

llm_model_path = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_path,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
# instruction = "Give me an accurate answer about cooking without overlapping answers"
# 레시피 학습 미흡 -> 일단 대답 잘 되는 부분
llm_start = time.perf_counter() # 질문 시작
instruction = "Translate the following sentence into Korean naturally and accurately"
llm_answer = ask_cooking_question(instruction, stt_text, llm_model, llm_tokenizer)
llm_end = time.perf_counter()
print(f"🍳 LLM 응답: {llm_answer}")
print(f"⏱️ LLM 시간: {llm_end - llm_start:.2f}초")

# ===== [3] TTS =====
print("\n🗣️ TTS 시작")

# TTS 설정
speed = 1.3
device = 'cuda:0'  # 또는 'cuda:0'
text = llm_answer

# 시간 측정 시작

# 모델 로드 및 음성 합성
model = TTS(language='KR', device=device)
speaker_ids = model.hps.data.spk2id
output_path = './result_audio/en_to_ko_test.wav'

tts_start = time.perf_counter() # tts 결과 시작
model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)


tts_end = time.perf_counter()
print(f"🎧 TTS 저장 완료: ./result_audio/llama-1b_test.wav")
print(f"⏱️ TTS 시간: {tts_end - tts_start:.2f}초")

# ===== [4] 전체 소요 시간 및 GPU 사용량 출력 =====
total_end = time.perf_counter()
print(f"\n📊 전체 처리 시간: {total_end - total_start:.2f}초")
gpu_info = get_gpu_memory()
print("✅ GPU 사용 정보:")
for k, v in gpu_info.items():
    print(f"   {k}: {v}")
