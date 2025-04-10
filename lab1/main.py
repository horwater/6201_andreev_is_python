import requests
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os
from dotenv import load_dotenv


# Загрузка переменных окружения из .env файла
load_dotenv()


class ImageProcessor:
    """Класс для обработки изображений животных"""

    # Конфигурация API (классовые атрибуты)
    API_URL = os.getenv('CAT_API_URL')
    OUTPUT_DIR = 'processed_images'
    KERNEL = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])  # Ядро для повышения резкости 3x3

    def __init__(self, image_url=None, breed_name=None, image_array=None):
        """Инициализация объекта ImageProcessor"""
        self._image_url = image_url
        self._breed_name = breed_name
        self._image_array = image_array

    def get_image_url(self):
        """Получить URL изображения"""
        return self._image_url

    def get_breed_name(self):
        """Получить название породы"""
        return self._breed_name

    def get_image_array(self):
        """Получить изображение в виде numpy массива"""
        return self._image_array

    @classmethod
    def fetch_images(cls, limit=1):
        """Получение нескольких изображений животных через API"""
        try:
            response = requests.get(
                cls.API_URL,
                params={
                    "has_breeds": 1,
                    "limit": limit,
                    "api_key": os.getenv('CAT_API_KEY')
                }
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                raise ValueError("No data received from API")

            processors = []
            for item in data:
                image_url = item['url']

                # Проверяем, есть ли информация о породе
                if 'breeds' not in item or not item['breeds']:
                    print("Warning: No breed information available, using 'unknown'")
                    breed_name = 'unknown'
                else:
                    breed_info = item['breeds'][0]
                    breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()

                processors.append(cls(image_url, breed_name))

            return processors
        except Exception as e:
            print(f"Error fetching images from API: {e}")
            raise

    def download_image(self):
        """Загрузка изображения по URL"""
        try:
            if not self._image_url:
                raise ValueError("No image URL set")

            response = requests.get(self._image_url, stream=True, timeout=10)
            response.raise_for_status()

            # Создаем временный файл для сохранения
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            temp_file = os.path.join(self.OUTPUT_DIR, "temp_image.jpg")

            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Загружаем изображение в numpy массив
            self._image_array = np.array(Image.open(temp_file))

            # Удаляем временный файл
            os.remove(temp_file)

            return True
        except Exception as e:
            print(f"Error downloading image: {e}")
            raise

    @staticmethod
    def _histogram_equalization(image):
        """Выравнивание гистограммы изображения (приватный метод)"""
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

    def _manual_convolution(self, image, kernel):
        """Ручная реализация свертки 3x3 (приватный метод)"""
        if len(image.shape) == 2:  # Grayscale
            result = self._convolve_channel(image, kernel)
        elif len(image.shape) == 3:  # RGB
            channels = []
            for i in range(3):
                channels.append(self._convolve_channel(image[:, :, i], kernel))
            result = np.stack(channels, axis=2)
        else:
            raise ValueError("Unsupported image dimensions")

        return np.clip(result, 0, 255).astype(np.uint8)

    def _convolve_channel(self, channel, kernel):
        """Свертка для одного канала (приватный метод)"""
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

    def manual_process(self):
        """Обработка изображения ручной сверткой"""
        if self._image_array is None:
            raise ValueError("No image loaded for processing")

        # 1. Ручная свертка
        manual_result = self._manual_convolution(self._image_array, self.KERNEL)

        # Применяем выравнивание гистограммы только для RGB изображений
        if len(manual_result.shape) == 3:
            manual_result = self._histogram_equalization(manual_result)

        return {'manual': manual_result}

    def scipy_process(self):
        """Обработка изображения с использованием scipy"""
        if self._image_array is None:
            raise ValueError("No image loaded for processing")

        # 2. Scipy свертка
        if len(self._image_array.shape) == 3:  # RGB
            scipy_result = np.zeros_like(self._image_array, dtype=np.float32)
            for i in range(3):
                scipy_result[:, :, i] = convolve(self._image_array[:, :, i].astype(float),
                                                 self.KERNEL, mode='reflect')
        else:  # серые тона
            scipy_result = convolve(self._image_array.astype(float), self.KERNEL, mode='reflect')

        scipy_result = np.clip(scipy_result, 0, 255).astype(np.uint8)

        return {'scipy': scipy_result}

    def save_images(self, index, processed_results, suffix):
        """Сохранение обработанных изображений"""
        if self._image_array is None:
            raise ValueError("No image loaded for saving")

        try:
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)

            # Определяем имя файла на основе типа обработки
            filename = os.path.join(
                self.OUTPUT_DIR,
                f"{index}_{self._breed_name}_{suffix}.jpg"
            )

            # Сохраняем обработанное изображение
            Image.fromarray(processed_results[list(processed_results.keys())[0]]).save(filename)

            return filename
        except Exception as e:
            print(f"Error saving images: {e}")
            raise


def main():
    """Основная функция выполнения программы"""
    try:
        # Получаем 3 изображения от API
        processors = ImageProcessor.fetch_images(limit=3)

        print(f"\nПолучено {len(processors)} изображений для обработки")

        results = []
        for i, processor in enumerate(processors, 1):
            print(f"\nОбработка изображения {i}...")

            # Загружаем изображение
            processor.download_image()

            # Сохраняем оригинал
            original_filename = os.path.join(
                processor.OUTPUT_DIR,
                f"{i}_{processor.get_breed_name()}_original.jpg"
            )
            Image.fromarray(processor.get_image_array()).save(original_filename)

            # Обрабатываем изображение
            processed_manual = processor.manual_process()
            processed_scipy = processor.scipy_process()

            # Сохраняем результаты
            saved_manual = processor.save_images(i, processed_manual, "manual")
            saved_scipy = processor.save_images(i, processed_scipy, "scipy")

            results.append({
                'original': original_filename,
                'manual': saved_manual,
                'scipy': saved_scipy
            })

            print(f"Изображение {i} сохранено как:")
            print(f"- Оригинал: {original_filename}")
            print(f"- Ручная обработка: {saved_manual}")
            print(f"- Scipy обработка: {saved_scipy}")

        print("\nВсе изображения успешно обработаны!")
        return 0
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        return 1


main()