from ultralytics import YOLO
import os

class IngredientsDetect():
    def __init__(self, path='/home/sungjin/capstone/test_img2'):
        self.path = path
        self.images = os.listdir(self.path)
        self.model = YOLO('best.pt')
        self.ingredient_dict = self._create_ingredient_dictionary()
    
    def _create_ingredient_dictionary(self):
        return {
            'bell pepper': '파프리카',
            'capsicum': '파프리카', 
            'bell pepper/capsicum': '파프리카',
            'broccoli': '브로콜리',
            'carrot': '당근',
            'cauliflower': '콜리플라워',
            'celery': '셀러리',
            'cucumber': '오이',
            'cuke': '오이',
            'cucumber/cuke': '오이',
            'lettuce': '상추',
            'onion': '양파',
            'potato': '감자',
            'tomato': '토마토',
            'chili': '고추',
            'egg plant': '가지',
            'eggplant': '가지',
            'paprika': '파프리카',
            'cabbage': '양배추',
            'garlic': '마늘',
            'leek': '부추',
            'sweet potato': '고구마',
            'mushroom': '버섯',
            'button mushroom': '양송이버섯',
            'oyster mushroom': '느타리버섯',
            'pork': '돼지고기',
            'beef': '소고기',
            'chicken meat': '닭고기',
            'chicken': '닭고기',
            'egg': '계란',
            'tofu': '두부',
            'bean curd': '두부',
        }
    
    def _get_image_path(self, idx):
        return os.path.join(self.path, self.images[idx])
    
    def _predict_ingredients(self, image_path):
        results = self.model.predict(
            source=image_path,
            conf=0.7,
            device=0,
            verbose=False
        )

        return results
    
    def _extract_class_names(self, results):
        detected_classes = set()
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = self.model.names[int(box.cls)]
                    detected_classes.add(class_name)
        
        return detected_classes
    
    def _translate_ingredients(self, detected_classes):
        translated_ingredients = []
        
        for ingredient in detected_classes:
            korean = self.ingredient_dict.get(ingredient.lower().strip())
            if korean:
                translated_ingredients.append(korean)
        
        return translated_ingredients
    
    def _format_result(self, translated_ingredients):
        if not translated_ingredients:
            return "알 수 없는 식재료가 검출되었습니다"
        
        unique_ingredients = sorted(set(translated_ingredients))
        return ', '.join(unique_ingredients)
    
    def __call__(self, idx):
        image_path = self._get_image_path(idx)
        results = self._predict_ingredients(image_path)
        detected_classes = self._extract_class_names(results)
        
        if not detected_classes:
            return "식재료가 검출되지 않았습니다"
        
        translated_ingredients = self._translate_ingredients(detected_classes)
        
        return self._format_result(translated_ingredients)