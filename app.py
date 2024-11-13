from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from PIL import Image
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Inicialización de CUDA y creación de un contexto global
drv.init()
device = drv.Device(0)
context = device.make_context()  # Crea un contexto persistente para toda la aplicación

mod = SourceModule("""
    __global__ void applyConvolutionGPU(unsigned char* d_image, double* d_kernel, double* d_result, int width, int height, int kernel_size) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int half_kernel = kernel_size / 2;

        if (x < width && y < height) {
            if (x >= half_kernel && x < width - half_kernel && y >= half_kernel && y < height - half_kernel) {
                double sum = 0.0;
                for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                    for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                        int pixel_value = d_image[(y + ky) * width + (x + kx)];
                        sum += pixel_value * d_kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
                    }
                }
                d_result[y * width + x] = sum;
            } else {
                d_result[y * width + x] = 0;
            }
        }
    }
""")

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

# def apply_filter(image, kernel, width, height, kernel_size, hilos):
#     dest = np.zeros_like(image, dtype=np.float64)
    
#     # Calcular block_size en función de la cantidad de hilos especificada
#     block_dim = int(np.sqrt(hilos))  # La raíz cuadrada para bloques de forma cuadrada
#     block_size = (block_dim, block_dim, 1)
#     grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])), 1)

#     # Asegurarse de que el contexto global esté activo
#     context.push()
#     try:
#         applyConvolutionGPU = mod.get_function("applyConvolutionGPU")
        
#         # Medir el tiempo de ejecución de la GPU
#         start_time = time.time()  # Tiempo inicial
        
#         applyConvolutionGPU(
#             drv.In(image), drv.In(kernel), drv.Out(dest), 
#             np.int32(width), np.int32(height), np.int32(kernel_size), 
#             block=block_size, grid=grid_size
#         )
        
#         context.synchronize()  # Asegura la sincronización de la GPU
#         end_time = time.time()  # Tiempo final
        
#         gpu_time = end_time - start_time  # Tiempo de ejecución en segundos
#         print(f"Tiempo de ejecución en la GPU: {gpu_time:.6f} segundos")  # Muestra el tiempo en la consola
        
#     finally:
#         context.pop()  # Libera el contexto después de la operación
    
#     min_val, max_val = np.min(dest), np.max(dest)
#     return ((dest - min_val) / (max_val - min_val) * 255).astype(np.uint8)
def apply_filter(image, kernel, width, height, kernel_size, hilos):
    dest = np.zeros_like(image, dtype=np.float64)
    
    block_dim = int(np.sqrt(hilos))
    block_size = (block_dim, block_dim, 1)
    grid_size = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])), 1)

    context.push()
    try:
        applyConvolutionGPU = mod.get_function("applyConvolutionGPU")
        
        start_time = time.time()
        
        applyConvolutionGPU(
            drv.In(image), drv.In(kernel), drv.Out(dest), 
            np.int32(width), np.int32(height), np.int32(kernel_size), 
            block=block_size, grid=grid_size
        )
        
        context.synchronize()
        end_time = time.time()
        gpu_time = end_time - start_time
        
    finally:
        context.pop()
    
    min_val, max_val = np.min(dest), np.max(dest)
    normalized_image = ((dest - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return {
        "filtered_image": normalized_image,
        "gpu_time": gpu_time,
        "kernel_size": kernel_size,
        "num_blocks": grid_size[0] * grid_size[1],
        "num_threads": block_dim**2,
        "image_size": (width, height)
    }


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})

#     file = request.files['file']
#     filter_type = request.form.get('filter_type')
#     kernel_size = int(request.form.get('kernel_size', 5))
#     hilos = int(request.form.get('hilos', 1024))  # Obtener cantidad de hilos de la solicitud

#     if file.filename == '':
#         return jsonify({"error": "No selected file"})

#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     gray_image, width, height = load_image(filepath)
    
#     # Selección del filtro en función del tipo solicitado
#     if filter_type == 'emboss':
#         kernel = create_emboss_kernel(kernel_size)
#     elif filter_type == 'gabor':
#         kernel = create_gabor_kernel(kernel_size, 4.0, 0, 10.0, 0.5, 0)
#     elif filter_type == 'high_boost':
#         kernel = create_high_boost_kernel(kernel_size, 10.0)
#     else:
#         return jsonify({"error": "Filter type not supported"})

#     try:
#         result = apply_filter(gray_image, kernel, width, height, kernel_size, hilos)
#     except drv.LogicError as e:
#         return jsonify({"error": "CUDA Error: " + str(e)})

#     result_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
#     save_image(result, width, height, result_filepath)

#     return jsonify({"original": filepath, "processed": result_filepath})

# def load_image(filename):
#     image = Image.open(filename).convert('L')
#     gray_image = np.array(image, dtype=np.uint8)
#     width, height = image.size
#     return gray_image, width, height

# def save_image(result, width, height, filename):
#     result_image = Image.fromarray(result)
#     result_image.save(filename)

# if __name__ == '__main__':
#     # Ejecuta el servidor Flask en modo de un solo hilo
#     app.run(debug=True, threaded=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    filter_type = request.form.get('filter_type')
    kernel_size = int(request.form.get('kernel_size', 5))
    hilos = int(request.form.get('hilos', 1024))

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
    else:
        return jsonify({"error": "Filter type not supported"})

    try:
        result_data = apply_filter(gray_image, kernel, width, height, kernel_size, hilos)
    except drv.LogicError as e:
        return jsonify({"error": "CUDA Error: " + str(e)})

    result_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
    save_image(result_data["filtered_image"], width, height, result_filepath)

    return jsonify({
        "original": filepath,
        "processed": result_filepath,
        "gpu_time": result_data["gpu_time"],
        "kernel_size": result_data["kernel_size"],
        "num_blocks": result_data["num_blocks"],
        "num_threads": result_data["num_threads"],
        "image_size": result_data["image_size"]
    })

def load_image(filename):
    image = Image.open(filename).convert('L')
    gray_image = np.array(image, dtype=np.uint8)
    width, height = image.size
    return gray_image, width, height

def save_image(result, width, height, filename):
    result_image = Image.fromarray(result)
    result_image.save(filename)

if __name__ == '__main__':
    # Ejecuta el servidor Flask en modo de un solo hilo
    app.run(debug=True, threaded=False)
