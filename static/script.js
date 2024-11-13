
// document.getElementById("uploadForm").onsubmit = async function(event) {
//     event.preventDefault();

//     let formData = new FormData(this);

//     const response = await fetch("/upload", {
//         method: "POST",
//         body: formData
//     });
//     const result = await response.json();
    
//     if (result.error) {
//         alert(result.error);
//         return;
//     }

//     alert("Imagen Cargada Correctamente");

//     document.getElementById("original").src = "/" + result.original;
//     document.getElementById("processed").src = "/" + result.processed;
//     document.getElementById("original").style.display = "block";
//     document.getElementById("processed").style.display = "block";

//     document.getElementById("Mascara").textContent = "Mascara Utilizada: " + result.kernel_size + " x " + result.kernel_size;
//     document.getElementById("bloques").textContent = "Número de Bloques: " + result.num_blocks;
//     document.getElementById("hilos").textContent = "Número de Hilos: " + result.num_threads;
//     document.getElementById("Procesamiento").textContent = "Tiempo de Procesamiento: " + result.gpu_time.toFixed(6) + " segundos";
//     document.getElementById("informacion").textContent = "Tamaño de la Imagen: " + result.image_size[0] + " x " + result.image_size[1];
    
//     document.querySelector(".resultado").style.display = "block";
//     document.getElementById("images").style.display = "block";
// };


document.getElementById("uploadForm").onsubmit = async function(event) {
    event.preventDefault();

    let formData = new FormData(this);
    
    // Verificar si el campo de hilos está vacío, y si lo está, asignar el valor 1024
    const hilosInput = document.getElementById("hilo");
    if (hilosInput.value === "") {
        hilosInput.value = 1024; // Asignar 1024 si está vacío
    }

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const result = await response.json();
    
    if (result.error) {
        alert(result.error);
        return;
    }

    alert("Imagen Cargada Correctamente");

    // Mostrar las imágenes y otra información en el HTML
    document.getElementById("original").src = "/" + result.original;
    document.getElementById("processed").src = "/" + result.processed;
    document.getElementById("original").style.display = "block";
    document.getElementById("processed").style.display = "block";

    document.getElementById("Mascara").textContent = "Mascara Utilizada: " + result.kernel_size + " x " + result.kernel_size;
    document.getElementById("bloques").textContent = "Número de Bloques: " + result.num_blocks;
    document.getElementById("Procesamiento").textContent = "Tiempo de Procesamiento: " + result.gpu_time.toFixed(6) + " segundos";
    document.getElementById("informacion").textContent = "Tamaño de la Imagen: " + result.image_size[0] + " x " + result.image_size[1];
    
    document.getElementById("hilos").textContent = "Número de Hilos: " + hilosInput.value;


    document.querySelector(".resultado").style.display = "block";
    document.getElementById("images").style.display = "block";
};
