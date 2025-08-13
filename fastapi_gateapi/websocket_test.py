import asyncio
import websockets
import os
import json

# ==================== 설정 ====================
# 서버 2의 주소
SERVER2_URI = "ws://203.252.240.40:8000/ws/chat" 
# 테스트에 사용할 음성 파일
INPUT_AUDIO_PATH = "./response_audio/pipeline_test.wav" 
# 최종 AI 음성 응답을 저장할 파일
OUTPUT_AUDIO_PATH = "./response_audio/final_bot_response.wav" 
# ============================================

async def run_e2e_test():
    """종단 간(End-to-End) WebSocket 통신을 테스트합니다."""
    
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"!!! 에러: 입력 오디오 파일 '{INPUT_AUDIO_PATH}'를 찾을 수 없습니다.")
        return

    try:
        # 파일 크기 제한 없이 연결
        async with websockets.connect(SERVER2_URI, max_size=None) as websocket:
            print(f"✅ 서버 2에 연결 성공: {SERVER2_URI}")

            # 1. 테스트 음성 파일 전송
            with open(INPUT_AUDIO_PATH, 'rb') as f_audio:
                audio_data = f_audio.read()
            
            print(f"🔊 '{INPUT_AUDIO_PATH}' 파일을 서버로 전송합니다...")
            await websocket.send(audio_data)
            print("▶️ 전송 완료. 서버의 응답을 기다립니다...")

            # 2. 서버로부터 응답(텍스트, 음성) 수신 및 처리
            bot_audio_response = bytearray()
            
            while True:
                message = await websocket.recv()

                if isinstance(message, str):
                    # 서버가 보낸 JSON 형식의 텍스트 메시지 처리
                    data = json.loads(message)
                    msg_type = data.get("type")
                    msg_data = data.get("data")

                    if msg_type == "user_text":
                        print("\n" + "="*20)
                        print(f"💬 (나의 질문) STT 결과: {msg_data}")
                        print("="*20)
                    elif msg_type == "bot_text":
                        print("\n" + "*"*20)
                        print(f"🤖 (AI 답변) LLM 결과: {msg_data}")
                        print("*"*20)
                        print("🎧 이제 AI 음성 응답 수신을 시작합니다...")
                    elif msg_type == "event" and msg_data == "TTS_STREAM_END":
                        print("\n✅ 모든 응답 수신 완료.")
                        break # 모든 과정이 끝났으므로 루프 종료
                
                elif isinstance(message, bytes):
                    # 수신한 음성 데이터를 bytearray에 추가
                    bot_audio_response.extend(message)

            # 3. 수신한 음성 데이터를 파일로 저장
            with open(OUTPUT_AUDIO_PATH, 'wb') as f_out:
                f_out.write(bot_audio_response)
            
            if bot_audio_response:
                print(f"🎶 성공! AI의 최종 음성 응답이 '{OUTPUT_AUDIO_PATH}' 파일로 저장되었습니다.")
            else:
                print("⚠️ 음성 응답 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ 테스트 중 에러 발생: {e}")

if __name__ == "__main__":
    asyncio.run(run_e2e_test())
