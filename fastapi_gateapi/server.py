# /CookDuck/backend-server/fastapi/fastapi_gateapi/server.py
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import httpx
import os
import uuid
import logging
import asyncio
import json

# --- 기본 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://203.252.240.65:8001")
logger.info(f"AI 서버 주소: {AI_SERVER_URL}")
TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

async def stream_audio(websocket: WebSocket, client: httpx.AsyncClient, audio_filename: str):
    """오디오를 스트리밍합니다."""
    async with client.stream("GET", f"{AI_SERVER_URL}/audio/{audio_filename}") as audio_response:
        audio_response.raise_for_status()
        async for chunk in audio_response.aiter_bytes():
            await websocket.send_bytes(chunk)
    
    await websocket.send_text(json.dumps({"type": "event", "data": "TTS_STREAM_END"}))
    logger.info(f"오디오 스트리밍 완료: {audio_filename}")

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("클라이언트와 WebSocket 연결 성공.")
    
    local_temp_files = []
    server1_response_files = []
    
    try:
        # --- 초기 인사말 전송 ---
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                greeting_info_response = await client.post(f"{AI_SERVER_URL}/generate-greeting")
                greeting_info_response.raise_for_status()
                greeting_info = greeting_info_response.json()
                greeting_text = greeting_info["llm_text"]
                audio_filename = greeting_info["audio_filename"]
                await websocket.send_text(json.dumps({"type": "bot_text", "data": greeting_text}))
                await stream_audio(websocket, client, audio_filename)
            logger.info("초기 인사말 처리 완료.")
        except Exception as e:
            logger.error(f"초기 인사말 처리 중 에러 발생: {e}")

        # --- 사용자 음성 요청 처리 루프 ---
        while True:
            # 1. 클라이언트 음성 수신 및 임시 파일 생성
            wav_data = await websocket.receive_bytes()
            uid = str(uuid.uuid4())
            wav_path = os.path.join(TEMP_DIR, f"{uid}.wav")
            local_temp_files.append(wav_path)
            with open(wav_path, 'wb') as f: f.write(wav_data)
            
            # 2. STT 요청 및 사용자 텍스트 우선 전송
            user_text = ""
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(wav_path, "rb") as f_wav:
                    files = {"audio": (f"{uid}.wav", f_wav, "audio/wav")}
                    stt_response = await client.post(f"{AI_SERVER_URL}/stt", files=files)
                stt_response.raise_for_status()
                user_text = stt_response.json()["text"]
                
                # [핵심] 사용자 텍스트를 클라이언트에 우선 전송
                await websocket.send_text(json.dumps({"type": "user_text", "data": user_text}))
                logger.info(f"사용자 텍스트 우선 전송 완료: {user_text}")
            
            # 3. LLM/TTS 요청 및 결과 전송
            async with httpx.AsyncClient(timeout=90.0) as client:
                payload = {"text": user_text}
                llm_info_response = await client.post(f"{AI_SERVER_URL}/generate-llm-response", json=payload)
                llm_info_response.raise_for_status()
                
                llm_info = llm_info_response.json()
                bot_response_text = llm_info["llm_text"]
                audio_filename = llm_info["audio_filename"]
                
                server1_response_files.append(audio_filename)
                
                # [핵심] 봇의 답변 텍스트를 나중에 전송
                await websocket.send_text(json.dumps({"type": "bot_text", "data": bot_response_text}))
                logger.info(f"봇 텍스트 전송 완료: {bot_response_text}")

                # 오디오 스트리밍
                await stream_audio(websocket, client, audio_filename)

    except WebSocketDisconnect:
        logger.info("클라이언트가 정상적으로 연결을 종료했습니다.")
    except Exception as e:
        logger.error(f"WebSocket 처리 중 예상치 못한 에러 발생: {e}")
    finally:
        # 세션 종료 시 모든 임시 파일 정리
        logger.info("세션 종료. 파일 정리를 시작합니다.")
        for file_path in local_temp_files:
            try:
                if os.path.exists(file_path): os.remove(file_path)
            except Exception as e:
                logger.error(f"로컬 임시 파일 삭제 실패 {file_path}: {e}")
        
        if server1_response_files:
            logger.info(f"서버 1의 응답 오디오 파일 {len(server1_response_files)}개 정리 시작...")
            try:
                async with httpx.AsyncClient() as client:
                    delete_tasks = [client.delete(f"{AI_SERVER_URL}/audio/{fname}") for fname in server1_response_files]
                    await asyncio.gather(*delete_tasks, return_exceptions=True)
                logger.info("서버 1 응답 오디오 파일 정리 완료.")
            except Exception as e:
                 logger.error(f"서버 1 파일 정리 중 에러 발생: {e}")
        
        logger.info("모든 세션 정리 완료.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
