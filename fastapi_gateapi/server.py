# # /CookDuck/backend-server/fastapi/fastapi_gateapi/server.py
# import uvicorn
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
# import httpx
# import os
# import uuid
# import logging
# import asyncio

# # ==================== 로깅 및 기본 설정 ====================
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# app = FastAPI()

# # ==================== 서버 1 (AI 서버) 주소 설정 ====================
# AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://203.252.240.65:8001")
# logger.info(f"AI 서버 주소: {AI_SERVER_URL}")

# TEMP_DIR = "/tmp"
# os.makedirs(TEMP_DIR, exist_ok=True)

# # ==================== 실시간 음성 채팅 WebSocket 엔드포인트 ====================
# @app.websocket("/ws/chat")
# async def websocket_chat_endpoint(websocket: WebSocket):
#     """
#     클라이언트와 WebSocket 연결을 맺고,
#     음성을 받아 AI 서버와 통신한 후, 음성으로 다시 응답합니다.
#     """
#     await websocket.accept()
#     logger.info("클라이언트와 WebSocket 연결 성공.")
    
#     try:
#         while True:
#             # 1. 클라이언트로부터 음성 데이터 수신
#             pcm_data = await websocket.receive_bytes()
            
#             uid = str(uuid.uuid4())
#             pcm_path = os.path.join(TEMP_DIR, f"{uid}.pcm")
#             wav_path = os.path.join(TEMP_DIR, f"{uid}.wav")

#             # 2. 수신한 PCM 데이터를 파일로 저장하고 WAV로 변환
#             with open(pcm_path, 'wb') as f:
#                 f.write(pcm_data)
            
#             logger.info(f"PCM 데이터 수신 완료: {pcm_path}")
            
#             process = await asyncio.create_subprocess_exec(
#                 "ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
#                 "-i", pcm_path, wav_path,
#                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
#             )
#             await process.communicate()

#             if process.returncode != 0:
#                 raise Exception("ffmpeg 오디오 변환 실패")

#             logger.info(f"WAV 파일 변환 성공: {wav_path}")

#             # 3. (서버1 호출) 변환된 WAV 파일을 STT API로 전송
#             async with httpx.AsyncClient(timeout=60.0) as client:
#                 with open(wav_path, "rb") as f_wav:
#                     files = {"audio": (f"{uid}.wav", f_wav, "audio/wav")}
#                     stt_response = await client.post(f"{AI_SERVER_URL}/stt", files=files)
#             stt_response.raise_for_status()
#             user_text = stt_response.json()["text"]
#             logger.info(f"STT 결과 수신: {user_text}")

#             # 4. (서버1 호출) STT 결과를 LLM/TTS API로 전송
#             async with httpx.AsyncClient(timeout=90.0) as client:
#                 payload = {"text": user_text}
#                 tts_response = await client.post(f"{AI_SERVER_URL}/generate-speech", json=payload)
#             tts_response.raise_for_status()
#             logger.info("LLM/TTS 음성 응답 수신 시작...")

#             # 5. (클라이언트로 전송) 서버1로부터 받은 음성 응답을 실시간으로 클라이언트에 전송
#             async for chunk in tts_response.aiter_bytes():
#                 await websocket.send_bytes(chunk)
            
#             logger.info("음성 응답 전송 완료.")
#             await websocket.send_text('{"event": "TTS_STREAM_END"}')

#     # ==================== [수정된 부분] ====================
#     except WebSocketDisconnect:
#         # 클라이언트가 정상적으로 연결을 끊었을 때의 처리
#         logger.info("클라이언트가 정상적으로 연결을 종료했습니다.")
#     except httpx.HTTPStatusError as e:
#         error_text = f"AI 서버 에러: {e.response.status_code} - {e.response.text}"
#         logger.error(error_text)
#         await websocket.send_text(f'{{"error": "{error_text}"}}')
#     except Exception as e:
#         # 그 외 예상치 못한 에러 처리
#         logger.error(f"WebSocket 처리 중 예상치 못한 에러 발생: {e}")
#         # 연결이 아직 살아있다면 클라이언트에 에러 메시지 전송
#         try:
#             await websocket.send_text(f'{{"error": "An unexpected error occurred: {e}"}}')
#         except:
#             pass
#     # =======================================================
#     finally:
#         # 임시 파일 정리
#         if 'pcm_path' in locals() and os.path.exists(pcm_path): os.remove(pcm_path)
#         if 'wav_path' in locals() and os.path.exists(wav_path): os.remove(wav_path)
# /CookDuck/backend-server/fastapi/fastapi_gateapi/server.py
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


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("클라이언트와 WebSocket 연결 성공.")

    # --- 연결 직후 초기 인사말 전송 ---
    try:
        logger.info("초기 인사말 생성 요청...")
        async with httpx.AsyncClient(timeout=90.0) as client:
            greeting_info_response = await client.post(f"{AI_SERVER_URL}/generate-greeting")
            greeting_info_response.raise_for_status()
            
            greeting_info = greeting_info_response.json()
            greeting_text = greeting_info["llm_text"]
            audio_filename = greeting_info["audio_filename"]
            
            # Flutter에 봇의 인사말 텍스트 전송
            await websocket.send_text(json.dumps({"type": "bot_text", "data": greeting_text}))
            
            # 실제 인사말 오디오 파일 스트리밍
            async with client.stream("GET", f"{AI_SERVER_URL}/audio/{audio_filename}") as audio_response:
                audio_response.raise_for_status()
                async for chunk in audio_response.aiter_bytes():
                    await websocket.send_bytes(chunk)
            
            await websocket.send_text(json.dumps({"type": "event", "data": "TTS_STREAM_END"}))
        logger.info("초기 인사말 전송 완료.")
    except Exception as e:
        logger.error(f"초기 인사말 처리 중 에러 발생: {e}")
    # --- 인사말 로직 끝 ---

    # --- 사용자 음성 요청 처리 루프 ---
    try:
        while True:
            # 1. 클라이언트 음성 수신 및 변환
            pcm_data = await websocket.receive_bytes()
            uid = str(uuid.uuid4())
            pcm_path = os.path.join(TEMP_DIR, f"{uid}.pcm")
            wav_path = os.path.join(TEMP_DIR, f"{uid}.wav")
            with open(pcm_path, 'wb') as f:
                f.write(pcm_data)
            
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
                "-i", pcm_path, wav_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            if process.returncode != 0:
                raise Exception("ffmpeg 오디오 변환 실패")

            # 2. STT 요청
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(wav_path, "rb") as f_wav:
                    files = {"audio": (f"{uid}.wav", f_wav, "audio/wav")}
                    stt_response = await client.post(f"{AI_SERVER_URL}/stt", files=files)
            stt_response.raise_for_status()
            user_text = stt_response.json()["text"]
            await websocket.send_text(json.dumps({"type": "user_text", "data": user_text}))
            
            # 3. LLM/TTS 요청 및 응답 전송
            async with httpx.AsyncClient(timeout=90.0) as client:
                payload = {"text": user_text}
                llm_info_response = await client.post(f"{AI_SERVER_URL}/generate-llm-response", json=payload)
                llm_info_response.raise_for_status()
                
                llm_info = llm_info_response.json()
                bot_response_text = llm_info["llm_text"]
                audio_filename = llm_info["audio_filename"]
                
                await websocket.send_text(json.dumps({"type": "bot_text", "data": bot_response_text}))
                
                async with client.stream("GET", f"{AI_SERVER_URL}/audio/{audio_filename}") as audio_response:
                    audio_response.raise_for_status()
                    async for chunk in audio_response.aiter_bytes():
                        await websocket.send_bytes(chunk)
            
            await websocket.send_text(json.dumps({"type": "event", "data": "TTS_STREAM_END"}))

    except WebSocketDisconnect:
        logger.info("클라이언트가 정상적으로 연결을 종료했습니다.")
    except Exception as e:
        logger.error(f"WebSocket 처리 중 예상치 못한 에러 발생: {e}")
    finally:
        # 임시 파일 정리
        if 'pcm_path' in locals() and os.path.exists(pcm_path): os.remove(pcm_path)
        if 'wav_path' in locals() and os.path.exists(wav_path): os.remove(wav_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

# uvicorn server:app --host 0.0.0.0 --port 8000

