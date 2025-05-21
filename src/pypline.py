import numpy as np
import os
from dotenv import load_dotenv

from multiprocessing import Pool, cpu_count
import aiohttp

from PIL import Image
import time
import datetime

from src.image_processor import ImageProcessor

load_dotenv()


class AsyncImagePipeline:
    """Async pipeline class with parallel processing"""

    API_URL = os.getenv("CAT_API_URL")
    OUTPUT_DIR = "processed_images"
    current_time = datetime.datetime.now()

    def __init__(self, limit=1):
        self.limit = limit
        self.data = None
        self.processor = ImageProcessor()

    async def fetch_data(self):
        """Асинхронное получение данных"""
        start_time = time.time()
        print(f"\nНачало загрузки метаданных для {self.limit} изображений...")

        async with aiohttp.ClientSession() as session:
            params = {
                "has_breeds": 1,
                "limit": self.limit,
                "api_key": os.getenv("CAT_API_KEY"),
            }
            async with session.get(self.API_URL, params=params) as response:
                response.raise_for_status()
                self.data = await response.json()

        elapsed = time.time() - start_time
        print(f"Метаданные загружены ({elapsed:.2f} сек)")
        return start_time

    async def fetch_images(self):
        """Асинхронный генератор для извлечения данных изображений"""
        if not self.data:
            await self.fetch_data()

        for idx, item in enumerate(self.data):
            print(f"Получена информация об изображении {idx + 1}/{self.limit}")
            yield {
                "idx": idx + 1,
                "image_url": item["url"],
                "breed_name": item["breeds"][0]["name"].replace(" ", "_").lower(),
            }

    async def download_image(self, img_gen):
        """Асинхронная загрузка изображений"""
        async for img_data in img_gen:
            start_time = time.time()
            print(
                f"\n{self.current_time} Начало загрузки изображения {img_data['idx']}..."
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(img_data["image_url"]) as response:
                    response.raise_for_status()
                    image_bytes = await response.read()

                # Создание временного файла
                os.makedirs(self.OUTPUT_DIR, exist_ok=True)
                temp_file = os.path.join(self.OUTPUT_DIR, f"temp_{img_data['idx']}.jpg")

                with open(temp_file, "wb") as f:
                    f.write(image_bytes)

                # Преобразование в numpy array
                image_array = np.array(Image.open(temp_file))
                os.remove(temp_file)

                elapsed = time.time() - start_time
                print(
                    f"{self.current_time} Изображение {img_data['idx']} загружено ({elapsed:.2f} сек, размер: {image_array.shape})"
                )

                yield {"image_array": image_array, **img_data}
    @staticmethod
    def process_wrapper(img_data):
        """Wrapper function for processing a single image (must be at class level)"""
        processor = ImageProcessor()
        return processor.process_single_image(img_data)

    async def process_images(self, img_gen):
        """Параллельная обработка изображений с правильным использованием multiprocessing.Pool"""
        # Собираем все изображения для обработки
        images_to_process = []
        async for img_data in img_gen:
            images_to_process.append(img_data)

        print(f"\nНачало параллельной обработки {len(images_to_process)} изображений на {cpu_count() // 2} ядрах...")
        start_time = time.time()

        # Обработка в пуле процессов

        with Pool(processes=cpu_count() // 2) as pool:
            results = pool.map(self.process_wrapper, images_to_process)

        elapsed = time.time() - start_time
        print(f"Все изображения обработаны параллельно ({elapsed:.2f} сек)")

        # Возвращаем результаты через генератор
        for result in results:
            yield result

    async def save_images(self, img_gen):
        """Асинхронное сохранение изображений"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        async for img_data in img_gen:
            idx = img_data["idx"]
            start_time = time.time()
            print(f"\nНачало сохранения изображения {idx}...")

            # Сохранение оригинала
            original_filename = os.path.join(
                self.OUTPUT_DIR,
                f"{idx}_{img_data['breed_name']}_original.jpg",
            )
            Image.fromarray(img_data["image_array"]).save(original_filename)

            # Сохранение ручного результата
            manual_filename = os.path.join(
                self.OUTPUT_DIR,
                f"{idx}_{img_data['breed_name']}_manual.jpg",
            )
            Image.fromarray(img_data["manual"]).save(manual_filename)

            # Сохранение результата через SciPy
            scipy_filename = os.path.join(
                self.OUTPUT_DIR, f"{idx}_{img_data['breed_name']}_scipy.jpg"
            )
            Image.fromarray(img_data["scipy"]).save(scipy_filename)

            elapsed = time.time() - start_time
            print(
                f"{self.current_time} Изображение {idx} сохранено ({elapsed:.2f} сек)"
            )

            yield {
                "original": original_filename,
                "manual": manual_filename,
                "scipy": scipy_filename,
            }

    async def run_pipeline(self):
        """Запуск всего асинхронного пайплайна"""
        total_start = time.time()
        print(f"Запуск пайплайна для {self.limit} изображений")

        save_gen = self.save_images(self.process_images(self.download_image(self.fetch_images())))

        # Итерируемся по финальному генератору
        async for final_data in save_gen:
            print(f"Готово: {final_data}")

        total_elapsed = time.time() - total_start
        print(f"\nВесь пайплайн завершен за {total_elapsed:.2f} секунд")
