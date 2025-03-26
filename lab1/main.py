import requests
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os

# API конфигурация
API_KEY = 'live_xJowRcD2GFmlEhuAjQkdZhH4axl47FGCfI1JL2KHRs1HgYyGUmt4y7GxV7NvLufg'
API_URL = 'https://api.thecatapi.com/v1/images/search'
HEADERS = {'x-api-key': API_KEY}
OUTPUT_DIR = 'output_images'  # Папка для сохранения результатов


def get_animal_image():
    # получение изображения по API ключу
    params = {
        'has_breeds':1
    }

    try:
        response = requests.get(API_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("No data received from API")

        image_url = data[0]['url']
        # добавлена проверка на то, что массив изобраения не равен 0
        breed_info = data[0].get('breeds', [{}])[0] if data[0].get('breeds') else {}
        breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()

        return image_url, breed_name

    except Exception as e:
        print(f"Error fetching image from API: {e}")
        raise


def download_image(url, filename):
    # загрузка изображения и сохранение его оригинала
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Создаем папку для изображений, если она не существует
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise


def custom_convolution(image, kernel):
    # ручная свертка без использования библиотек
    if len(image.shape) == 2:  # Grayscale
        return _convolve_2d(image, kernel)
    elif len(image.shape) == 3:  # RGB
        channels = []
        for i in range(3):
            channels.append(_convolve_2d(image[:, :, i], kernel))
        return np.stack(channels, axis=2)
    else:
        raise ValueError("Unsupported image dimensions")


def _convolve_2d(image, kernel):
    # функция 2д свертки для одного канала (проходит по 3м каналам для rgb)
    # переворот ядра для свертки
    kernel = np.flipud(np.fliplr(kernel))

    # Pad the image
    pad_size = kernel.shape[0] // 2
    # создание боковых полей изображения (резервные пиксели)
    padded = np.pad(image, pad_size, mode='reflect')

    # подготовка выходного массива
    output = np.zeros_like(image)

    # суть свертка
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            output[y, x] = np.sum(region * kernel)

    # обрезка значений до допустимых
    return np.clip(output, 0, 255).astype(np.uint8)


def process_images(image_path, breed_name):
    # загрузка изображения
    img = Image.open(image_path)
    img_array = np.array(img)

    # ядро для повышения резкости
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ]) / 5.0

    # процесс безбиблиотечной свертки
    custom_result = custom_convolution(img_array, sharpening_kernel)
    custom_filename = os.path.join(OUTPUT_DIR, f"custom_sharpened_{breed_name}.jpg")
    Image.fromarray(custom_result).save(custom_filename)

    # процесс свертки scipy
    if len(img_array.shape) == 3:  # RGB проверка
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


def main():
    try:
        # Создаем папку для выходных файлов
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # получение изображения
        image_url, breed_name = get_animal_image()
        original_filename = os.path.join(OUTPUT_DIR, f"original_{breed_name}.jpg")
        download_image(image_url, original_filename)

        # свертка и сохранение изображения
        results = process_images(original_filename, breed_name)

        print("\nProcessing complete. Saved files:")
        print(f"- Original: {results['original']}")
        print(f"- Custom convolution: {results['custom']}")
        print(f"- SciPy convolution: {results['scipy']}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Clean up partially downloaded files if any
        if 'original_filename' in locals() and os.path.exists(original_filename):
            os.remove(original_filename)


if __name__ == "__main__":
    main()