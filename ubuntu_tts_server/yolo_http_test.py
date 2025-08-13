import httpx
import asyncio
import os

# --- 설정 ---
# 실행 중인 YOLO 서버의 주소
# 만약 다른 컴퓨터에서 서버를 실행 중이라면, 해당 컴퓨터의 IP 주소로 변경하세요.
SERVER_URL = "http://127.0.0.1:8002/predict"

# 서버로 보낼 테스트 이미지 파일의 경로
# 실제 이미지 파일 경로로 수정해야 합니다.
IMAGE_PATH = "./yolo/salad_picture/20.png" 

async def call_predict_api(image_path: str):
    """
    YOLO 서버의 /predict API를 호출하여 이미지 속 재료를 분석합니다.

    Args:
        image_path (str): 분석할 이미지 파일의 경로.
    """
    if not os.path.exists(image_path):
        print(f"❌ 오류: 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요: {image_path}")
        return

    print(f"🚀 YOLO 서버에 분석을 요청합니다...")
    print(f"   - 대상 서버: {SERVER_URL}")
    print(f"   - 전송할 이미지: {image_path}")

    try:
        # 비동기 HTTP 클라이언트 생성
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 파일을 multipart/form-data 형식으로 열어서 준비
            with open(image_path, "rb") as f:
                # 'files' 파라미터는 튜플 형식으로 (파일명, 파일 객체, 컨텐츠 타입)을 전달할 수 있습니다.
                # FastAPI 서버의 'image: UploadFile = File(...)' 부분의 'image'와 키 이름을 맞춰야 합니다.
                files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
                
                # 서버에 POST 요청 보내기
                response = await client.post(SERVER_URL, files=files)

            # 서버 응답 상태 코드 확인 (200이 아니면 에러 발생)
            response.raise_for_status()

            # 성공적인 응답(JSON)을 파싱하여 결과 출력
            detected_ingredients = response.json()
            
            print("\n✅ 분석 성공! 서버로부터 아래의 재료 목록을 받았습니다:")
            print(f"   >>> {detected_ingredients}")

    except httpx.ConnectError as e:
        print(f"\n❌ 연결 실패: YOLO 서버가 실행 중인지 확인해주세요. (주소: {e.request.url})")
    except httpx.HTTPStatusError as e:
        # 서버에서 4xx 또는 5xx 오류를 반환한 경우
        print(f"\n❌ 서버 오류 발생: {e.response.status_code}")
        print(f"   - 오류 내용: {e.response.text}")
    except Exception as e:
        print(f"\n❌ 알 수 없는 오류가 발생했습니다: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 비동기 함수 실행
    asyncio.run(call_predict_api(IMAGE_PATH))
