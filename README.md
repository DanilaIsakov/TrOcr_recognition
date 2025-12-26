# Проект распознавания рукописного русского текста

Проект для распознавания рукописного русского текста на основе специализированной модели **cyrillic-trocr/trocr-handwritten-cyrillic** - TrOCR модели, специально обученной для кириллического рукописного текста.

## Структура проекта

```
OCR_project/
├── images/          # Папка с изображениями рукописного текста
├── writings/        # Папка с расшифровками (ground truth) для каждого изображения
├── recognize.py     # Скрипт для распознавания с готовыми моделями 
├── config.py        # Конфигурация проекта
├── requirements.txt # Зависимости проекта
└── README.md        # Документация проекта
```

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Убедитесь, что у вас установлен PyTorch (с поддержкой CUDA, если доступна видеокарта):
```bash
# Для CPU
pip install torch torchvision

# Для CUDA (пример для CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## Использование готовых моделей

Простой скрипт `recognize.py` позволяет использовать несколько готовых моделей:

### Распознавание с помощью TrOCR Cyrillic

Специализированная модель `cyrillic-trocr/trocr-handwritten-cyrillic` специально обучена для кириллического рукописного текста и дает лучшие результаты.

**Важно:** По умолчанию изображения автоматически сегментируются на строки перед распознаванием для повышения точности:

```bash
# Одно изображение (по умолчанию используется trocr-cyrillic с сегментацией)
python recognize.py --image images/DSCN4528.JPG

    # Или явно указать метод
    python recognize.py --image images/DSCN4528.JPG --method trocr-cyrillic

    # С использованием GPU (если доступно)
    python recognize.py --image images/DSCN4528.JPG --method trocr-cyrillic --gpu

    # Отключить сегментацию (распознавать все изображение целиком)
    python recognize.py --image images/DSCN4528.JPG --method trocr-cyrillic --no-segment
```

**Примечание:** При первом запуске модель скачается автоматически (~500MB).


## Формат данных

- **Изображения**: JPG, PNG, JPEG файлы в папке `images/`
- **Расшифровки**: TXT файлы в папке `writings/` с тем же именем, что и соответствующее изображение

Пример:
- `images/DSCN4528.JPG` → `writings/DSCN4528.txt`

Каждый текстовый файл содержит полный текст, который написан на соответствующем изображении (несколько строк).

## Модель

Проект использует специализированную модель **[cyrillic-trocr/trocr-handwritten-cyrillic](https://huggingface.co/cyrillic-trocr/trocr-handwritten-cyrillic)** - TrOCR модель, специально обученную для распознавания кириллического рукописного текста. 

Эта модель основана на [TrOCR (Transformer-based OCR)](https://github.com/microsoft/unilm/tree/master/trocr) от Microsoft и дообучена на датасете кириллического рукописного текста, что обеспечивает лучшие результаты для русского языка по сравнению с общей моделью.
