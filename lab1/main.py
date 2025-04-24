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
    API_URL = os.getenv("CAT_API_URL")
    OUTPUT_DIR = "processed_images"
    KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def __init__(self, limit=1):
        """Инициализация"""
        self.limit = limit
        self.image_data = {}
        response = requests.get(
            self.API_URL,
            params={
                "has_breeds": 1,
                "limit": self.limit,
                "api_key": os.getenv("CAT_API_KEY"),
            },
        )
        response.raise_for_status()
        self.data = response.json()

    def fetch_images(self):
        """Генератор извлечения данных изображений по API"""
        for idx, item in enumerate(self.data):
            print(f"fetching {idx + 1}")
            yield {
                "idx": idx + 1,
                "image_url": item["url"],
                "breed_name": item["breeds"][0]["name"].replace(" ", "_").lower(),
            }

    def download_image(self, gen):
        """Метод загрузки."""
        for img_data in gen:
            response = requests.get(img_data["image_url"], stream=True, timeout=10)
            response.raise_for_status()

            # Создание временного файла для сохранения
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            temp_file = os.path.join(self.OUTPUT_DIR, "temp_image.jpg")

            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Преобразование изображения в numpy-массив
            image_array = np.array(Image.open(temp_file))

            # Удаление временного файла
            os.remove(temp_file)
            print("numpy download done")

            yield {"image_array": image_array, **img_data}

    @staticmethod
    @njit
    def _convolve_channel(channel, kernel, padded):
        """Свертка для одного канала"""
        output = np.zeros_like(channel, dtype=np.float32)
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                region = padded[y : y + 3, x : x + 3]
                output[y, x] = np.sum(region * kernel)
        return output

    def process_image(self, gen):
        """Генерация обработанных изображений"""
        for img_data in gen:
            # 1. Ручная обработка
            manual_result = next(self._manual_convolution(img_data["image_array"], self.KERNEL))

            # 2. Обработка с помощью SciPy
            scipy_result = next(self._scipy_convolution(img_data["image_array"]))
            print("processed")

            yield {"manual": manual_result, "scipy": scipy_result, **img_data}

    def _manual_convolution(self, image, kernel):
        """Ручная реализация свертки"""
        if len(image.shape) == 2:  # grayscale
            pad_size = kernel.shape[0] // 2
            padded = np.pad(image, pad_size, mode="reflect")
            result = self._convolve_channel(image, kernel, padded)
        elif len(image.shape) == 3:  # RGB
            channels = []
            for i in range(3):
                pad_size = kernel.shape[0] // 2
                padded = np.pad(image[:, :, i], pad_size, mode="reflect")
                channels.append(self._convolve_channel(image[:, :, i], kernel, padded))
            result = np.stack(channels, axis=2)
        else:
            raise ValueError("Unsupported image dimensions")
        yield np.clip(result, 0, 255).astype(np.uint8)

    def _scipy_convolution(self, image):
        """Обработка с помощью SciPy"""
        if len(image.shape) == 3:  # RGB
            channels = [
                convolve(image[:, :, i].astype(float), self.KERNEL, mode="reflect")
                for i in range(3)
            ]
            result = np.stack(channels, axis=-1)
        else:  # grayscale
            result = convolve(image.astype(float), self.KERNEL, mode="reflect")
        yield np.clip(result, 0, 255).astype(np.uint8)

    def save_images(self, gen):
        """Генератор, сохраняющий исходное и обработанные изображения"""
        for img_data in gen:
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)

            # Сохранение оригинала
            original_filename = os.path.join(
                self.OUTPUT_DIR, f"{img_data['idx']}_{img_data['breed_name']}_original.jpg"
            )
            Image.fromarray(img_data["image_array"]).save(original_filename)

            # Сохранение ручного результата
            manual_filename = os.path.join(
                self.OUTPUT_DIR, f"{img_data['idx']}_{img_data['breed_name']}_manual.jpg"
            )
            Image.fromarray(img_data["manual"]).save(manual_filename)

            # Сохранение результата через SciPy
            scipy_filename = os.path.join(
                self.OUTPUT_DIR, f"{img_data['idx']}_{img_data['breed_name']}_scipy.jpg"
            )
            Image.fromarray(img_data["scipy"]).save(scipy_filename)
            print("saved.")
            yield {
                "original": original_filename,
                "manual": manual_filename,
                "scipy": scipy_filename,
            }

    def run_pipeline(self):
        """Метод запуска всего пайплайна."""
        save_gen = self.save_images(self.process_image(self.download_image(self.fetch_images())))

        # Итерация по последнему генератору
        for final_data in save_gen:
            print(final_data)


pipeline = ImagePipeline(limit=4)
pipeline.run_pipeline()