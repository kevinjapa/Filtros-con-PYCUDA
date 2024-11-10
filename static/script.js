document.getElementById("uploadForm").onsubmit = async function(event) {
    event.preventDefault();

    let formData = new FormData(this);

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });
    const result = await response.json();
    alert("Imagen Cargada Correctamente");
    if (result.error) {
        alert(result.error);
        return;
    }

    document.getElementById("original").src = "/" + result.original;
    document.getElementById("processed").src = "/" + result.processed;
    document.getElementById("original").style.display = "block";
    document.getElementById("processed").style.display = "block";

    const kernelInput = document.getElementById("kernel_size");
    const kernelOutput = document.getElementById("Mascara");
    kernelOutput.textContent = "Mascara Utilizada: "+kernelInput.value;

};

// document.addEventListener("DOMContentLoaded", function() {
//     const kernelInput = document.getElementById("kernel_size");
//     const kernelOutput = document.getElementById("Mascara");

//     kernelInput.addEventListener("input",function(){
//         kernelOutput.textContent = "Mascara Utilizada: "+kernelInput.value;

//     });
// });

