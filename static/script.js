// document.getElementById("uploadForm").onsubmit = async function(event) {
//     event.preventDefault();

//     let formData = new FormData(this);

//     const response = await fetch("/upload", {
//         method: "POST",
//         body: formData
//     });
//     const result = await response.json();
//     alert("Imagen Cargada Correctamente");
//     if (result.error) {
//         alert(result.error);
//         return;
//     }

//     document.getElementById("original").src = "/" + result.original;
//     document.getElementById("processed").src = "/" + result.processed;
//     document.getElementById("original").style.display = "block";
//     document.getElementById("processed").style.display = "block";

//     const kernelInput = document.getElementById("kernel_size");
//     const kernelOutput = document.getElementById("Mascara");
//     kernelOutput.textContent = "Mascara Utilizada: "+kernelInput.value + " X " + kernelInput.value;

//     // Mostrar los resultados
//     document.querySelector(".titulos").style.display = "block";
//     document.querySelector(".resultado").style.display = "block";
//     document.getElementById("kernel-size-display").textContent = result.kernel_size;
//     document.getElementById("bloques").textContent = "Número de Bloques: " + result.bloques;
//     document.getElementById("hilos").textContent = "Número de Hilos: " + result.hilos;
//     document.getElementById("Procesamiento").textContent = "Tiempo de Procesamiento: " + result.procesamiento;
//     document.getElementById("images").style.display = "block";

// };

// document.addEventListener("DOMContentLoaded", function() {
//     // const kernelInput = document.getElementById("kernel_size");
//     // const kernelOutput = document.getElementById("Mascara");

//     // kernelInput.addEventListener("input",function(){
//     //     kernelOutput.textContent = "Mascara Utilizada: "+kernelInput.value;

//     // });
//     // document.getElementById("original").textContent = " ";
// });

document.getElementById("uploadForm").onsubmit = async function(event) {
    event.preventDefault();

    let formData = new FormData(this);

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

    document.getElementById("original").src = "/" + result.original;
    document.getElementById("processed").src = "/" + result.processed;
    document.getElementById("original").style.display = "block";
    document.getElementById("processed").style.display = "block";

    document.getElementById("Mascara").textContent = "Mascara Utilizada: " + result.kernel_size + " x " + result.kernel_size;
    document.getElementById("bloques").textContent = "Número de Bloques: " + result.num_blocks;
    document.getElementById("hilos").textContent = "Número de Hilos: " + result.num_threads;
    document.getElementById("Procesamiento").textContent = "Tiempo de Procesamiento: " + result.gpu_time.toFixed(6) + " segundos";
    document.getElementById("informacion").textContent = "Tamaño de la Imagen: " + result.image_size[0] + " x " + result.image_size[1];
    
    document.querySelector(".resultado").style.display = "block";
    document.getElementById("images").style.display = "block";
};


