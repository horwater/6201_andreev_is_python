import requests
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os
from dotenv import load_dotenv
from numba import njit

# Загрузка переменных окружения из .env файла
load_dotenv()


class ImagePipeline:
    """Класс конвейера"""

    # Конфигурация API (классовые атрибуты)
    API_URL = os.getenv('CAT_API_URL')
    OUTPUT_DIR = 'processed_images'
    KERNEL = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])  # Ядро для повышения резкости 3x3

    def __init__(self, limit=1):
        """Инициализация"""
        self.limit = limit
        self.image_data = {}
        response = requests.get(
            self.API_URL,
            params={
                "has_breeds": 1,
                "limit": self.limit,
                "api_key": os.getenv('CAT_API_KEY')
            }
        )
        response.raise_for_status()
        self.data = response.json()

    def fetch_images(self):
        """Генератор извлечения данных изображений по API"""
        for idx, item in enumerate(self.data):
            yield {
                'idx': idx + 1,
                'image_url': item['url'],
                'breed_name': item['breeds'][0]['name'].replace(' ', '_').lower(),
            }

    def download_image(self, img_data):
        """Генератор, скачивающий изображение и возвращающий numpy-массив"""
        response = requests.get(img_data['image_url'], stream=True, timeout=10)
        response.raise_for_status()

        # Создание временного файла для сохранения
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        temp_file = os.path.join(self.OUTPUT_DIR, "temp_image.jpg")

        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Преобразование изображения в numpy-массив
        image_array = np.array(Image.open(temp_file))

        # Удаление временного файла
        os.remove(temp_file)

        yield {'image_array': image_array, **img_data}

    @staticmethod
    @njit
    def _apply_kernel_njit(region, kernel):
        """Применение ядра свертки с использованием Numba"""
        return np.sum(region * kernel)

    def process_image(self, img_data):
        """Генератор, применяющий обработку изображения вручную и через SciPy"""
        if img_data['image_array'] is None:
            raise ValueError("Нет загруженного изображения для обработки")

        # 1. Скалярное повышение резкости вручную с использованием Numba
        manual_result = next(self._manual_convolution(img_data['image_array'], self.KERNEL))

        # 2. Использование библиотеки SciPy для свёртки
        scipy_result = next(self._scipy_convolution(img_data['image_array']))

        yield {'manual': manual_result, 'scipy': scipy_result, **img_data}

    def _manual_convolution(self, image, kernel):
        """Ручная реализация свертки 3x3 (приватный метод) с Numba"""
        if len(image.shape) == 2:  # Grayscale
            result = next(self._convolve_channel(image, kernel))
        elif len(image.shape) == 3:  # RGB
            channels = []
            for i in range(3):
                channels.append(next(self._convolve_channel(image[:, :, i], kernel)))
            result = np.stack(channels, axis=2)
        else:
            raise ValueError("Unsupported image dimensions")

        yield np.clip(result, 0, 255).astype(np.uint8)

    def _convolve_channel(self, channel, kernel):
        """Свертка для одного канала (приватный метод) с Numba"""
        # Подготовка выходного массива
        output = np.zeros_like(channel, dtype=np.float32)

        # Добавление padding
        pad_size = kernel.shape[0] // 2
        padded = np.pad(channel, pad_size, mode='reflect')

        # Применение свертки с использованием Numba
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                region = padded[y:y + 3, x:x + 3]
                output[y, x] = self._apply_kernel_njit(region, kernel)

        yield output

    def _scipy_convolution(self, image):
        """Использование SciPy для повышения резкости"""
        if len(image.shape) == 3:  # RGB-изображение
            channels = []
            for i in range(3):
                channels.append(convolve(image[:, :, i].astype(float), self.KERNEL, mode='reflect'))
            result = np.stack(channels, axis=-1)
        else:  # Серое изображение
            result = convolve(image.astype(float), self.KERNEL, mode='reflect')
        yield np.clip(result, 0, 255).astype(np.uint8)

    def save_images(self, img_data):
        """Генератор, сохраняющий исходное и обработанные изображения"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Сохранение оригинального изображения
        original_filename = os.path.join(
            self.OUTPUT_DIR,
            f"{img_data['idx']}_{img_data['breed_name']}_original.jpg"
        )
        Image.fromarray(img_data['image_array']).save(original_filename)

        # Сохранение результатов обработки
        manual_filename = os.path.join(
            self.OUTPUT_DIR,
            f"{img_data['idx']}_{img_data['breed_name']}_manual.jpg"
        )
        Image.fromarray(img_data['manual']).save(manual_filename)

        scipy_filename = os.path.join(
            self.OUTPUT_DIR,
            f"{img_data['idx']}_{img_data['breed_name']}_scipy.jpg"
        )
        Image.fromarray(img_data['scipy']).save(scipy_filename)

        yield {
            'original': original_filename,
            'manual': manual_filename,
            'scipy': scipy_filename
        }

    def run_pipeline(self):
        """Запуск пайплайна обработки изображений"""
        for img_data in self.fetch_images():
            # Использование next() как в вашем примере
            saved = next(self.save_images(
                next(self.process_image(
                    next(self.download_image(img_data))
                ))
            ))
            print(f"Обработано и сохранено: {saved}")


pipeline = ImagePipeline(limit=3)
pipeline.run_pipeline()