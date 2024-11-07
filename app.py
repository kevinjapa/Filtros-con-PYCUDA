from flask import Flask, render_template, request, redirect, url_for, jsonify # type: ignore
import os
import numpy as np
# import pycuda.driver as drv
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
from PIL import Image
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# mod = SourceModule("""
#     __global__ void applyConvolutionGPU(unsigned char* d_image, double* d_kernel, double* d_result, int width, int height, int kernel_size) {
#         int x = blockIdx.x * blockDim.x + threadIdx.x;
#         int y = blockIdx.y * blockDim.y + threadIdx.y;
#         int half_kernel = kernel_size / 2;

#         if (x < width && y < height) {
#             if (x >= half_kernel && x < width - half_kernel && y >= half_kernel && y < height - half_kernel) {
#                 double sum = 0.0;
#                 for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
#                     for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
#                         int pixel_value = d_image[(y + ky) * width + (x + kx)];
#                         sum += pixel_value * d_kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
#                     }
#                 }
#                 d_result[y * width + x] = sum;
#             } else {
#                 d_result[y * width + x] = 0;
#             }
#         }
#     }
# """)

def create_emboss_kernel(kernel_size):
    kernel = np.zeros(kernel_size * kernel_size, dtype=np.float64)
    half_size = kernel_size // 2
    for y in range(kernel_size):
        for x in range(kernel_size):
            if x < half_size and y < half_size:
                kernel[y * kernel_size + x] = -1
            elif x > half_size and y > half_size:
                kernel[y * kernel_size + x] = 1
            elif x == half_size and y == half_size:
                kernel[y * kernel_size + x] = 1
    return kernel

def create_gabor_kernel(kernel_size, sigma, theta, lambda_, gamma, psi):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    half_size = kernel_size // 2
    for y in range(-half_size, half_size + 1):
        for x in range(-half_size, half_size + 1):
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            gauss = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
            sinusoid = np.cos(2 * np.pi * x_theta / lambda_ + psi)
            kernel[y + half_size, x + half_size] = gauss * sinusoid
    return kernel.flatten()

def create_high_boost_kernel(kernel_size, A):
    kernel = -np.ones(kernel_size * kernel_size, dtype=np.float64)
    half_size = kernel_size // 2
    kernel[half_size * kernel_size + half_size] = A + (kernel_size * kernel_size) - 1
    return kernel

def apply_filter(image, kernel, width, height, kernel_size):
    dest = np.zeros_like(image, dtype=np.float64)
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])), 1)
    # applyConvolutionGPU = mod.get_function("applyConvolutionGPU")
    # applyConvolutionGPU(drv.In(image), drv.In(kernel), drv.Out(dest), np.int32(width), np.int32(height), np.int32(kernel_size), block=block_size, grid=grid_size)
    # drv.Context.synchronize()
    min_val, max_val = np.min(dest), np.max(dest)
    return ((dest - min_val) / (max_val - min_val) * 255).astype(np.uint8)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    filter_type = request.form.get('filter_type')
    kernel_size = int(request.form.get('kernel_size'))

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    gray_image, width, height = load_image(filepath)
    if filter_type == 'emboss':
        kernel = create_emboss_kernel(kernel_size)
    elif filter_type == 'gabor':
        kernel = create_gabor_kernel(kernel_size, 4.0, 0, 10.0, 0.5, 0)
    elif filter_type == 'high_boost':
        kernel = create_high_boost_kernel(kernel_size, 10.0)

    result = apply_filter(gray_image, kernel, width, height, kernel_size)
    result_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
    save_image(result, width, height, result_filepath)

    return jsonify({"original": filepath, "processed": result_filepath})

def load_image(filename):
    image = Image.open(filename).convert('L')
    gray_image = np.array(image, dtype=np.uint8)
    width, height = image.size
    return gray_image, width, height

def save_image(result, width, height, filename):
    result_image = Image.fromarray(result)
    result_image.save(filename)

if __name__ == '__main__':
    app.run(debug=True)
