import numpy as np
from scipy.ndimage import convolve
from numba import njit
import time

class ImageProcessor:
    """Class for image processing operations"""
    start_time = time.time()
    KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    @staticmethod
    @njit
    def _convolve_channel(channel, kernel, padded):
        """Convolution for a single channel"""
        output = np.zeros_like(channel, dtype=np.float32)
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                region = padded[y: y + 3, x: x + 3]
                output[y, x] = np.sum(region * kernel)
        return output

    def process_single_image(self, img_data):
        """Process an image with both manual and scipy convolution"""
        idx = img_data["idx"]

        image_array = img_data["image_array"]

        # Manual convolution
        if len(image_array.shape) == 2:  # grayscale
            pad_size = self.KERNEL.shape[0] // 2
            padded = np.pad(image_array, pad_size, mode="reflect")
            manual_result = self._convolve_channel(image_array, self.KERNEL, padded)
        else:  # RGB
            channels = []
            for i in range(3):
                pad_size = self.KERNEL.shape[0] // 2
                padded = np.pad(image_array[:, :, i], pad_size, mode="reflect")
                channels.append(
                    self._convolve_channel(image_array[:, :, i], self.KERNEL, padded)
                )
            manual_result = np.stack(channels, axis=2)
        manual_result = np.clip(manual_result, 0, 255).astype(np.uint8)

        # SciPy convolution
        if len(image_array.shape) == 3:  # RGB
            channels = [
                convolve(
                    image_array[:, :, i].astype(float), self.KERNEL, mode="reflect"
                )
                for i in range(3)
            ]
            scipy_result = np.stack(channels, axis=-1)
        else:  # grayscale
            scipy_result = convolve(
                image_array.astype(float), self.KERNEL, mode="reflect"
            )
        scipy_result = np.clip(scipy_result, 0, 255).astype(np.uint8)

        elapsed = time.time() - self.start_time
        print(f"Изображение {idx} обработано ({elapsed:.2f} сек)")

        return {"manual": manual_result, "scipy": scipy_result, **img_data}