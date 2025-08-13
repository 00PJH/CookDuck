from ultralytics import YOLO
import os

def predict_and_translate_ingredients() -> dict:
    """
    이미지 디렉토리에서 재료를 탐지하고, 영문 클래스 이름을 한글로 번역하여
    딕셔너리 형태로 반환합니다.

    Returns:
        dict: 탐지된 재료의 딕셔너리 (예: {"egg": "계란", "Paprika": "파프리카"})
    """
    # 1. 영문 클래스 이름과 한글 이름 매핑 데이터 정의
    # 모델이 학습한 클래스 이름과 그에 맞는 한글 이름을 미리 정의해야 합니다.
    translation_map = {
        "Button mushroom": "양송이버섯",
        "egg": "계란",
        "Paprika": "파프리카",
        "Tomato": "토마토",
        "Lettuce": "양상추",
        "Cucumber": "오이",
        # 필요에 따라 다른 재료들을 추가할 수 있습니다.
    }

    # 2. 모델 로드 및 이미지 경로 설정
    base_path = './salad_picture'
    try:
        img_list = os.listdir(base_path)
    except FileNotFoundError:
        print(f"오류: '{base_path}' 디렉토리를 찾을 수 없습니다.")
        return {}

    model = YOLO('./best.pt')

    # 3. 모든 이미지에서 탐지된 재료를 중복 없이 저장하기 위한 집합(set)
    all_unique_classes = set()

    # 4. 각 이미지에 대해 예측 수행
    for img_name in img_list:
        # 이미지 전체 경로 생성 (기존 코드의 경로 중첩 오류 수정)
        img_path = os.path.join(base_path, img_name)
        
        # 파일이 맞는지, 이미지가 맞는지 추가 확인
        if not os.path.isfile(img_path):
            continue

        print(f"'{img_path}' 분석 중...")
        
        results = model.predict(
            source=img_path,
            imgsz=640,
            conf=0.7
        )

        # 5. 한 이미지 내에서 탐지된 클래스 이름 추출
        detected_classes_in_image = set()
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = model.names[int(box.cls)]
                    detected_classes_in_image.add(class_name)
        
        # 6. 전체 재료 목록에 추가
        all_unique_classes.update(detected_classes_in_image)

    # 7. 최종 결과를 딕셔너리 형태로 가공
    final_result = {}
    ingredient_list = []
    for class_name in all_unique_classes:
        # 매핑 데이터에 있는 클래스 이름인 경우, 한글 이름으로 변환하여 추가
        if class_name in translation_map:
            final_result[class_name] = translation_map[class_name]
            ingredient_list.append(translation_map[class_name])
        else:
            # 매핑 데이터에 없으면, 영문 이름 그대로 사용 (선택적)
            final_result[class_name] = class_name 
            print(f"경고: '{class_name}'에 대한 한글 번역이 translation_map에 없습니다.")
    


    return ingredient_list

# # --- 함수 실행 및 결과 출력 예시 ---
# if __name__ == '__main__':
#     detected_ingredients = predict_and_translate_ingredients()
    
#     if detected_ingredients:
#         print("\n--- 최종 탐지된 재료 목록 ---")
#         print(detected_ingredients)
#         # 예쁘게 출력
#         for eng, kor in detected_ingredients.items():
#             print(f"- {kor}")
# print(predict_and_translate_ingredients())