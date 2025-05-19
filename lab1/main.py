import aiohttp
import asyncio
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os
from dotenv import load_dotenv
from numba import njit
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime

# Загрузка ременных окружения
load_dotenv()


class AsyncImagePipeline:
    """Асинхронный класс конвейера с параллельной обработкой"""

    API_URL = os.getenv("CAT_API_URL")
    OUTPUT_DIR = "processed_images"
    KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def __init__(self, limit=1):
        """Инициализация"""
        self.limit = limit
        self.data = None

    async def fetch_data(self):
        """Асинхронное получение данных"""
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚡ Начало загрузки метаданных для {self.limit} изображений...")
        
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Метаданные загружены ({elapsed:.2f} сек)")

    async def fetch_images(self):
        """Асинхронный генератор для извлечения данных изображений"""
        if not self.data:
            await self.fetch_data()

        for idx, item in enumerate(self.data):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🖼️ Получена информация об изображении {idx + 1}/{self.limit}")
            yield {
                "idx": idx + 1,
                "image_url": item["url"],
                "breed_name": item["breeds"][0]["name"].replace(" ", "_").lower(),
            }

    async def download_image(self, img_gen):
        """Асинхронная загрузка изображений"""
        async for img_data in img_gen:
            start_time = time.time()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⬇️ Начало загрузки изображения {img_data['idx']}...")
            
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
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Изображение {img_data['idx']} загружено ({elapsed:.2f} сек, размер: {image_array.shape})")
                
                yield {"image_array": image_array, **img_data}

    @staticmethod
    def _convolve_channel(channel, kernel, padded):
        """Свертка для одного канала"""
        output = np.zeros_like(channel, dtype=np.float32)
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                region = padded[y : y + 3, x : x + 3]
                output[y, x] = np.sum(region * kernel)
        return output

    def _process_single_image(self, img_data):
        """Обработка одного изображения (для многопроцессорной обработки)"""
        idx = img_data["idx"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Начало обработки изображения {idx} в процессе {os.getpid()}...")
        start_time = time.time()
        
        image_array = img_data["image_array"]
        
        # Ручная свертка
        if len(image_array.shape) == 2:  # grayscale
            pad_size = self.KERNEL.shape[0] // 2
            padded = np.pad(image_array, pad_size, mode="reflect")
            manual_result = self._convolve_channel(image_array, self.KERNEL, padded)
        else:  # RGB
            channels = []
            for i in range(3):
                pad_size = self.KERNEL.shape[0] // 2
                padded = np.pad(image_array[:, :, i], pad_size, mode="reflect")
                channels.append(self._convolve_channel(image_array[:, :, i], self.KERNEL, padded))
            manual_result = np.stack(channels, axis=2)
        manual_result = np.clip(manual_result, 0, 255).astype(np.uint8)

        # SciPy свертка
        if len(image_array.shape) == 3:  # RGB
            channels = [
                convolve(image_array[:, :, i].astype(float), self.KERNEL, mode="reflect")
                for i in range(3)
            ]
            scipy_result = np.stack(channels, axis=-1)
        else:  # grayscale
            scipy_result = convolve(image_array.astype(float), self.KERNEL, mode="reflect")
        scipy_result = np.clip(scipy_result, 0, 255).astype(np.uint8)
        
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Изображение {idx} обработано ({elapsed:.2f} сек)")
        
        return {"manual": manual_result, "scipy": scipy_result, **img_data}

    async def process_images(self, img_gen):
        """Параллельная обработка изображений с использованием multiprocessing"""
        # Собираем все изображения для параллельной обработки
        images_to_process = []
        async for img_data in img_gen:
            images_to_process.append(img_data)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 Начало параллельной обработки {len(images_to_process)} изображений на {cpu_count()} ядрах...")
        start_time = time.time()
        
        # Обработка в пуле процессов
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self._process_single_image, images_to_process)
        
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Все изображения обработаны параллельно ({elapsed:.2f} сек)")

        # Возвращаем результаты через генератор
        for result in results:
            yield result

    async def save_images(self, img_gen):
        """Асинхронное сохранение изображений"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        async for img_data in img_gen:
            idx = img_data["idx"]
            start_time = time.time()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 💾 Начало сохранения изображения {idx}...")
            
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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Изображение {idx} сохранено ({elapsed:.2f} сек)")
            
            yield {
                "original": original_filename,
                "manual": manual_filename,
                "scipy": scipy_filename,
            }

    async def run_pipeline(self):
        """Запуск всего асинхронного пайплайна"""
        total_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏁 Запуск пайплайна для {self.limit} изображений")
        
        # Создаем асинхронный генератор
        img_gen = self.fetch_images()
        download_gen = self.download_image(img_gen)
        process_gen = self.process_images(download_gen)
        save_gen = self.save_images(process_gen)

        # Итерируемся по финальному генератору
        async for result in save_gen:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 Готово: {result}")

        total_elapsed = time.time() - total_start
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🌟 Весь пайплайн завершен за {total_elapsed:.2f} секунд")


async def main():
    pipeline = AsyncImagePipeline(limit=4)
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())