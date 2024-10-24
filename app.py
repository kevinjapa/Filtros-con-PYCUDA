from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
from source.FiltrosGpu import apply_filter, create_emboss_kernel

app = Flask(__name__)

# Directorios para las imágenes
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Crear carpetas si no existen
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Aplicar filtro Emboss con CUDA
    kernel_size = 21
    emboss_kernel = create_emboss_kernel(kernel_size)
    processed_image_path = apply_filter(filepath, emboss_kernel, kernel_size)

    # Mostrar las imágenes
    return render_template('result.html', 
                           original_image=file.filename, 
                           processed_image=os.path.basename(processed_image_path))


if __name__ == '__main__':
    app.run(debug=True)

