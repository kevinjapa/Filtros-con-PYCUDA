FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Instalación de dependencias
RUN apt-get -qq update && \
    apt-get -qq install -y build-essential python3-pip && \
    pip3 install pycuda flask numpy pillow

# Copia de los archivos de la aplicación
COPY . /app
WORKDIR /app

# Exponer el puerto 5000 para Flask
EXPOSE 5000

# Comando para iniciar el servidor Flask
CMD ["python3", "app.py"]