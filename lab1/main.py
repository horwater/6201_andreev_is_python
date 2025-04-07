import requests
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Конфигурация API
API_URL = os.getenv('CAT_API_URL')
OUTPUT_DIR = 'processed_images'
KERNEL = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])  # Ядро для повышения резкости 3x3


def get_animal_image():
    """Получение изображения животного через API"""
    try:
        response = requests.get(
            API_URL,
            params={
                "has_breeds": 1,
                "limit": 1,
                "api_key": os.getenv('CAT_API_KEY')
            }
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("No data received from API")

        image_url = data[0]['url']

        # Проверяем, есть ли информация о породе
        if 'breeds' not in data[0] or not data[0]['breeds']:
            print("Warning: No breed information available, using 'unknown'")
            breed_name = 'unknown'
        else:
            breed_info = data[0]['breeds'][0]
            breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()

        return image_url, breed_name
    except Exception as e:
        print(f"Error fetching image from API: {e}")
        raise


def imageHGistogram(image):
    """Выравнивание гистограммы изображения"""
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Функция работает только с RGB изображениями")

    # Копируем массив, чтобы не изменять оригинал
    result = image.copy().astype(np.float32)

    for i in range(3):
        channel = result[:, :, i]
        min_val = np.min(channel)
        max_val = np.max(channel)

        if max_val - min_val > 0:
            result[:, :, i] = 255 * (channel - min_val) / (max_val - min_val)
        else:
            result[:, :, i] = channel

    return np.clip(result, 0, 255).astype(np.uint8)


def download_image(url, filename):
    """Загрузка и сохранение изображения"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise


def manual_convolution(image, kernel):
    """Ручная реализация свертки 3x3"""
    if len(image.shape) == 2:  # Grayscale
        result = _convolve_channel(image, kernel)
    elif len(image.shape) == 3:  # RGB
        channels = []
        for i in range(3):
            channels.append(_convolve_channel(image[:, :, i], kernel))
        result = np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image dimensions")

    return np.clip(result, 0, 255).astype(np.uint8)


def _convolve_channel(channel, kernel):
    """Свертка для одного канала"""
    # Подготовка выходного массива
    output = np.zeros_like(channel, dtype=np.float32)

    # Добавление padding
    pad_size = kernel.shape[0] // 2
    padded = np.pad(channel, pad_size, mode='reflect')

    # Применение свертки
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            region = padded[y:y + 3, x:x + 3]
            output[y, x] = np.sum(region * kernel)

    return output


def process_image(image_path, breed_name):
    """Обработка изображения разными методами свертки"""
    try:
        # Загрузка изображения
        img = Image.open(image_path)
        img_array = np.array(img)

        # 1. Ручная свертка
        manual_result = manual_convolution(img_array, KERNEL)

        # Применяем выравнивание гистограммы только для RGB изображений
        if len(manual_result.shape) == 3:
            manual_result = imageHGistogram(manual_result)

        manual_filename = os.path.join(OUTPUT_DIR, f"manual_{breed_name}.jpg")
        Image.fromarray(manual_result).save(manual_filename)

        # 2. Scipy свертка
        if len(img_array.shape) == 3:  # RGB
            scipy_result = np.zeros_like(img_array, dtype=np.float32)
            for i in range(3):
                scipy_result[:, :, i] = convolve(img_array[:, :, i].astype(float),
                                               KERNEL, mode='reflect')
        else:  # серые тона
            scipy_result = convolve(img_array.astype(float), KERNEL, mode='reflect')

        scipy_result = np.clip(scipy_result, 0, 255).astype(np.uint8)
        scipy_filename = os.path.join(OUTPUT_DIR, f"scipy_{breed_name}.jpg")
        Image.fromarray(scipy_result).save(scipy_filename)

        return {
            'original': image_path,
            'manual': manual_filename,
            'scipy': scipy_filename
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        raise


def main():
    """Основная функция выполнения программы"""
    try:
        # Создание директории для результатов
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Получение изображения
        image_url, breed_name = get_animal_image()
        original_filename = os.path.join(OUTPUT_DIR, f"original_{breed_name}.jpg")
        download_image(image_url, original_filename)

        # Обработка изображения
        results = process_image(original_filename, breed_name)

        # Вывод результатов
        print("\nОбработка завершена. Сохраненные файлы:")
        print(f"- Исходное изображение: {results['original']}")
        print(f"- Ручная свертка: {results['manual']}")
        print(f"- SciPy свертка: {results['scipy']}")

    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        return 1

    return 0


main()