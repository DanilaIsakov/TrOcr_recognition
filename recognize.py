import os
import argparse
from PIL import Image
import cv2
import numpy as np


def preprocess_line_image(image):
    """
    Предобработка изображения строки для улучшения распознавания
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image с улучшенным контрастом и нормализацией
    """
    img_array = np.array(image.convert('RGB'))
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    enhanced = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    
    mean_intensity = np.mean(enhanced)
    if mean_intensity < 127:
        enhanced = cv2.bitwise_not(enhanced)
    
    min_height = 32
    if enhanced.shape[0] < min_height:
        scale_factor = min_height / enhanced.shape[0]
        new_width = int(enhanced.shape[1] * scale_factor)
        enhanced = cv2.resize(enhanced, (new_width, min_height), interpolation=cv2.INTER_CUBIC)
    
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(enhanced_rgb)


def recognize_with_trocr_cyrillic(image_path, model_path=None, device="cpu", segment_lines=True, segment_method='auto'):
    """Распознавание текста с помощью специализированной TrOCR модели для кириллицы"""
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from config import Config
        
        print("Загрузка TrOCR модели для кириллического рукописного текста...")
        
        if model_path and os.path.exists(model_path):
            processor = TrOCRProcessor.from_pretrained(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
        else:
            model_name = Config.MODEL_NAME
            print(f"Используется модель: {model_name}")
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        model.eval()
        model = model.to(device)
        
        print(f"Распознавание текста на изображении: {image_path}")
        
        original_image_path = image_path
        
        if segment_lines:
            print("Сегментация изображения на строки...")
            # Здесь предполагается, что сегментация строки была бы добавлена в другом месте
            # Вместо этого, можно обработать изображение как одно целое (если сегментация не требуется)
            line_images = [Image.open(image_path).convert('RGB')]  # Загружаем одно изображение
            print(f"Найдено строк: 1")
            
            recognized_lines = []
            for i, line_image in enumerate(line_images, 1):
                try:
                    width, height = line_image.size
                    if width < 10 or height < 10:
                        print(f"  Строка {i}/{len(line_images)}: пропущена (слишком маленькая: {width}x{height})")
                        continue
                    
                    print(f"  Распознавание строки {i}/{len(line_images)} (размер: {width}x{height})...")
                    
                    line_image = preprocess_line_image(line_image)
                    
                    pixel_values = processor(line_image, return_tensors="pt").pixel_values.to(device)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=Config.MAX_LENGTH,
                            num_beams=6,  
                            early_stopping=True,
                            length_penalty=1.2,  
                            repetition_penalty=1.3,  
                            no_repeat_ngram_size=3,  
                            temperature=0.7,  
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id
                        )
                    
                    line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if line_text.strip(): 
                        recognized_lines.append(line_text)
                        print(f"    Результат: {line_text[:50]}...")
                except Exception as e:
                    print(f"  Ошибка при распознавании строки {i}/{len(line_images)}: {e}")
                    continue
            
            generated_text = '\n'.join(recognized_lines) if recognized_lines else ""
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                if image.size[0] < 10 or image.size[1] < 10:
                    print("Предупреждение: изображение слишком маленькое для распознавания")
                    return ""
                
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=Config.MAX_LENGTH,
                        num_beams=6,
                        early_stopping=True,
                        length_penalty=1.2,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=3,
                        temperature=0.7,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except Exception as e:
                print(f"Ошибка при распознавании изображения: {e}")
                return ""
        
        return generated_text.strip() if generated_text else ""
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите необходимые зависимости: pip install transformers torch opencv-python")
        return None
    except Exception as e:
        print(f"Ошибка при распознавании с TrOCR Cyrillic: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Распознавание рукописного русского текста с использованием готовых моделей"
    )
    parser.add_argument("--image", type=str, default=None, help="Путь к изображению")
    parser.add_argument("--method", type=str, default="trocr-cyrillic", 
                       choices=["trocr", "trocr-cyrillic"],
                       help="Метод распознавания (по умолчанию: trocr-cyrillic - рекомендуется для русского)")
    parser.add_argument("--batch", type=str, default=None,
                       help="Папка с изображениями для пакетной обработки")
    parser.add_argument("--output", type=str, default=None,
                       help="Путь к файлу для сохранения результатов")
    parser.add_argument("--gpu", action="store_true",
                       help="Использовать GPU (если доступен)")
    parser.add_argument("--trocr-model", type=str, default=None,
                       help="Путь к обученной TrOCR модели")
    parser.add_argument("--no-segment", action="store_true",
                       help="Отключить сегментацию по строкам (распознавать все изображение целиком)")
    
    args = parser.parse_args()
    
    device = "cuda" if args.gpu else "cpu"
    
    if not args.image and not args.batch:
        parser.error("Необходимо указать либо --image, либо --batch")
    
    results = []
    
    if args.image and os.path.isfile(args.image):
        images_to_process = [args.image]
    elif args.batch and os.path.isdir(args.batch):
        images_to_process = [
            os.path.join(args.batch, f) 
            for f in sorted(os.listdir(args.batch))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))
        ]
    else:
        print("Ошибка: не указан корректный путь к изображению или папке")
        return
    
    for img_path in images_to_process:
        print(f"\n{'='*60}")
        print(f"Обработка: {os.path.basename(img_path)}")
        print(f"Метод: {args.method.upper()}")
        print(f"{'='*60}")
        
        segment = not args.no_segment
        
        if args.method == "trocr-cyrillic":
            text = recognize_with_trocr_cyrillic(img_path, model_path=args.trocr_model, device=device, 
                                                 segment_lines=segment)
        else:
            print(f"Неизвестный метод: {args.method}")
            continue
        
        if text:
            try:
                print(f"\nРаспознанный текст:\n{text}\n")
            except UnicodeEncodeError:
                print("\nРаспознанный текст:")
                print(text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                print()
            results.append((os.path.basename(img_path), text))
        else:
            print("Не удалось распознать текст")
    
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            for img_name, text in results:
                f.write(f"{'='*60}\n")
                f.write(f"Изображение: {img_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{text}\n\n")
        print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
