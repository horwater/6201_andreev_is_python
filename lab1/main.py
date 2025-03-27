import requests
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Конфигурация из .env
API_KEY = os.getenv('CAT_API_KEY', '')
API_URL = os.getenv('CAT_API_URL', 'https://api.thecatapi.com/v1/images/search')
HEADERS = {'x-api-key': API_KEY} if API_KEY else {}
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output_images')


def get_animal_image():
    """Получение изображения животного через API"""
    params = {'has_breeds': 1, 'limit': 1}

    try:
        response = requests.get(API_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("No data received from API")

        image_url = data[0]['url']
        breed_info = data[0].get('breeds', [{}])[0] if data[0].get('breeds') else {}
        breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()

        return image_url, breed_name
    except Exception as e:
        print(f"Error fetching image from API: {e}")
        raise


def download_image(url, filename):
    """Загрузка и сохранение изображения"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise


def custom_convolution(image, kernel):
    """Ручная реализация свертки"""
    if len(image.shape) == 2:  # Grayscale
        result = _convolve_2d(image, kernel)
    elif len(image.shape) == 3:  # RGB
        channels = []
        for i in range(3):
            channels.append(_convolve_2d(image[:, :, i], kernel))
        result = np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image dimensions")

    return np.clip(result, 0, 255).astype(np.uint8)


def _convolve_2d(image, kernel):
    """2D свертка для одного канала"""
    kernel = np.flipud(np.fliplr(kernel))
    pad_size = kernel.shape[0] // 2
    padded = np.pad(image, pad_size, mode='reflect')
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            output[y, x] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


def process_images(image_path, breed_name):
    """Обработка изображения с разными методами свертки"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        # Ядро для повышения резкости 5x5
        sharpening_kernel = np.array([
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]
        ])

        # 1. Ручная свертка
        custom_result = custom_convolution(img_array, sharpening_kernel)
        custom_filename = os.path.join(OUTPUT_DIR, f"custom_sharpened_{breed_name}.jpg")
        Image.fromarray(custom_result).save(custom_filename)

        # 2. Scipy свертка
        if len(img_array.shape) == 3:  # RGB
            scipy_result = np.zeros_like(img_array, dtype=np.float32)
            for i in range(3):
                scipy_result[:, :, i] = convolve(img_array[:, :, i].astype(float),
                                               sharpening_kernel,
                                               mode='reflect')
        else:  # серые тона
            scipy_result = convolve(img_array.astype(float), sharpening_kernel, mode='reflect')

        scipy_result = np.clip(scipy_result, 0, 255).astype(np.uint8)
        scipy_filename = os.path.join(OUTPUT_DIR, f"scipy_sharpened_{breed_name}.jpg")
        Image.fromarray(scipy_result).save(scipy_filename)

        return {
            'original': image_path,
            'custom': custom_filename,
            'scipy': scipy_filename
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        raise


def main():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        image_url, breed_name = get_animal_image()
        original_filename = os.path.join(OUTPUT_DIR, f"original_{breed_name}.jpg")
        download_image(image_url, original_filename)

        results = process_images(original_filename, breed_name)

        print("\nProcessing complete. Saved files:")
        print(f"- Original: {results['original']}")
        print(f"- Custom convolution: {results['custom']}")
        print(f"- SciPy convolution: {results['scipy']}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if 'original_filename' in locals() and os.path.exists(original_filename):
            os.remove(original_filename)


if __name__ == "__main__":
    main()