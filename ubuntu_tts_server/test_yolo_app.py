import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import shutil
from typing import List

# --- 기본 설정 ---
# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 임시 이미지 파일을 저장할 디렉토리 생성
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# --- YOLO 모델 및 번역 데이터 로드 ---
# 모델을 전역 변수로 선언하여 서버 시작 시 한 번만 로드하도록 함
yolo_model = None
translation_map = {
    "Button mushroom": "양송이버섯",
    "egg": "계란",
    "Paprika": "파프리카",
    "Tomato": "토마토",
    "Lettuce": "양상추",
    "Cucumber": "오이",
    # 필요에 따라 다른 재료들을 추가할 수 있습니다.
}

@app.on_event("startup")
def load_model():
    """서버가 시작될 때 YOLO 모델을 로드합니다."""
    global yolo_model
    try:
        yolo_model = YOLO('./yolo/best.pt')
        print("✅ YOLO 모델 로딩 성공.")
    except Exception as e:
        print(f"❌ YOLO 모델 로딩 실패: {e}")
        # 모델 로딩 실패는 심각한 문제이므로, 여기서는 서버를 종료하거나
        # 상태를 '비정상'으로 관리할 수 있습니다.
        # 간단한 예제이므로 우선 print만 합니다.

# --- API 엔드포인트 정의 ---
@app.post("/predict", response_model=List[str])
async def predict_ingredients_from_image(image: UploadFile = File(...)):
    """
    이미지 파일을 업로드받아 재료를 탐지하고,
    탐지된 재료의 한글 이름 리스트를 반환합니다.
    """
    if not yolo_model:
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")

    # 업로드된 이미지 파일을 임시로 저장
    temp_file_path = os.path.join(TEMP_IMAGE_DIR, image.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        print(f"'{temp_file_path}' 분석 중...")
        
        # YOLO 모델로 예측 수행
        results = yolo_model.predict(
            source=temp_file_path,
            imgsz=640,
            conf=0.7,
            verbose=False # 터미널에 상세 로그를 출력하지 않음
        )

        # 탐지된 모든 클래스 이름을 중복 없이 저장
        detected_classes = set()
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = yolo_model.names[int(box.cls)]
                    detected_classes.add(class_name)

        # 최종 결과를 한글 이름 리스트로 변환
        ingredient_list = []
        for class_name in detected_classes:
            if class_name in translation_map:
                ingredient_list.append(translation_map[class_name])
            else:
                print(f"경고: '{class_name}'에 대한 한글 번역이 translation_map에 없습니다.")
                ingredient_list.append(class_name) # 번역이 없으면 영문명 추가

        print(f"탐지된 재료: {ingredient_list}")
        return JSONResponse(content=ingredient_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")
    finally:
        # 처리 완료 후 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- 서버 실행 ---
if __name__ == "__main__":
    # 예를 들어 8002번 포트에서 서버를 실행합니다.
    uvicorn.run(app, host="0.0.0.0", port=8002)
