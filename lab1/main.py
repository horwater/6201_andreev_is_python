import requests
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Конфигурация API
API_KEY = os.getenv('CAT_API_KEY', '')
API_URL = os.getenv('CAT_API_URL', 'https://api.thecatapi.com/v1/images/search')
HEADERS = {'x-api-key': API_KEY} if API_KEY else {}
OUTPUT_DIR = 'processed_images'
KERNEL = np.array([[0, -1, 0],
                   [-1, 3, -1],
                   [0, -1, 0]])  # Ядро для повышения резкости 3x3


def get_animal_image():
    """Получение изображения животного через API"""
    try:
        response = requests.get(API_URL, headers=HEADERS, params={'has_breeds': True, 'limit': 1}, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("No data received from API")

        image_url = data[0]['url']
        breed_info = data[0].get('breeds', [{}])[0]
        breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()

        return image_url, breed_name
    except Exception as e:
        print(f"Error fetching image from API: {e}")
        raise


def imageHGistogram(image):
    min_red = np.min(image[:, :, 0])
    max_red = np.max(image[:, :, 0])
    min_green = np.min(image[:, :, 1])
    max_green = np.max(image[:, :, 1])
    min_blue = np.min(image[:, :, 2])
    max_blue = np.max(image[:, :, 2])

    k_red = 255 / (max_red - min_red)
    k_green = 255 / (max_green - min_green)
    k_blue = 255 / (max_blue - min_blue)

    b_red = 0 - (k_red * min_red)
    b_green = 0 - (k_red * min_red)
    b_blue = 0 - (k_red * min_red)

    image[:, :, 0] = k_red * image[:, :, 0] + b_red
    image[:, :, 1] = k_green * image[:, :, 1] + b_green
    image[:, :, 2] = k_blue * image[:, :, 2] + b_blue
    return image

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
    output = np.zeros_like(channel)

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
        manual_results_post =imageHGistogram(manual_result)
        manual_filename = os.path.join(OUTPUT_DIR, f"manual_{breed_name}.jpg")
        Image.fromarray(manual_results_post).save(manual_filename)

        # 2. Scipy свертка
        if len(img_array.shape) == 3:  # RGB
            scipy_result = np.zeros_like(img_array)
            for i in range(3):
                scipy_result[:, :, i] = convolve(img_array[:, :, i], KERNEL, mode='reflect')
        else:  # Grayscale
            scipy_result = convolve(img_array, KERNEL, mode='reflect')

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


if __name__ == "__main__":
    exit(main())