import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image
import os

# Kernel CUDA en formato de string
kernel_code = """
__global__ void applyConvolutionGPU(unsigned char* d_image, double* d_kernel, double* d_result, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_kernel = kernel_size / 2;

    if (x >= half_kernel && x < width - half_kernel && y >= half_kernel && y < height - half_kernel) {
        double sum = 0.0;
        for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
            for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                int pixel_value = d_image[(y + ky) * width + (x + kx)];
                sum += pixel_value * d_kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
            }
        }
        d_result[y * width + x] = sum;
    }
}
"""

mod = SourceModule(kernel_code)
apply_convolution_gpu = mod.get_function("applyConvolutionGPU")

# Funciones para crear kernels
def create_emboss_kernel(kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    half_size = kernel_size // 2
    for y in range(kernel_size):
        for x in range(kernel_size):
            if x < half_size and y < half_size:
                kernel[y, x] = -1
            elif x > half_size and y > half_size:
                kernel[y, x] = 1
            elif x == half_size and y == half_size:
                kernel[y, x] = 1
    return kernel.flatten()

# Función para aplicar el filtro usando CUDA
def apply_filter(image_path, kernel, kernel_size):
    # Cargar la imagen
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    width, height = img.size
    img_data = np.array(img, dtype=np.uint8)

    # Reservar memoria en GPU
    d_image = cuda.mem_alloc(img_data.nbytes)
    d_kernel = cuda.mem_alloc(kernel.nbytes)
    d_result = cuda.mem_alloc(img_data.nbytes)

    # Copiar datos a la GPU
    cuda.memcpy_htod(d_image, img_data)
    cuda.memcpy_htod(d_kernel, kernel)

    # Configurar los bloques e hilos para la GPU
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])), 1)

    # Ejecutar el kernel CUDA
    apply_convolution_gpu(d_image, d_kernel, d_result, np.int32(width), np.int32(height), np.int32(kernel_size), block=block_size, grid=grid_size)
    
    # Copiar el resultado de la GPU a la CPU
    result_data = np.empty_like(img_data, dtype=np.float64)
    cuda.memcpy_dtoh(result_data, d_result)

    # Normalizar y convertir a imagen
    result_data = np.clip(result_data, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_data)

    # Guardar la imagen procesada
    processed_image_path = os.path.splitext(image_path)[0] + '_processed.jpg'
    result_img.save(processed_image_path)

    # Liberar memoria GPU
    d_image.free()
    d_kernel.free()
    d_result.free()

    return processed_image_path
