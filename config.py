"""
Конфигурация для проекта распознавания рукописного русского текста
"""
import os

class Config:
    # Пути к данным
    IMAGES_DIR = "images"
    WRITINGS_DIR = "writings"
    
    # Параметры модели
    MODEL_NAME = "cyrillic-trocr/trocr-handwritten-cyrillic"  # Специализированная модель для кириллического рукописного текста
    MAX_LENGTH = 512  # Максимальная длина последовательности
    IMAGE_SIZE = (384, 384)  # Размер входного изображения

    RECOMMENDED_METHOD = "cyrillic-trocr/trocr-handwritten-cyrillic" 
    
    
    # Путь для сохранения модели
    OUTPUT_DIR = "model_checkpoints"
    
    # Разделение данных
    TRAIN_SPLIT = 0.8  # 80% на обучение, 20% на валидацию
    
    # Воспроизводимость
    SEED = 42

