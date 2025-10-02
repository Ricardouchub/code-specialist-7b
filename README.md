# Code Specialist 7B

Esta es una aplicación de chat local construida con Streamlit para interactuar y evaluar el modelo `Code-Specialist-7b`.

Este modelo es un fine-tuning de `mistralai/Mistral-7B-Instruct-v0.3` especializado en la generación de código Python y SQL, con un enfoque en tareas de ciencia y análisis de datos.

-   **Modelo Base:** `mistralai/Mistral-7B-Instruct-v0.3`
-   **Técnica de Fine-Tuning:** Supervised Fine-Tuning (SFT) utilizando adaptadores LoRA.
-   **Dataset:** Entrenado sobre un subconjunto filtrado de aproximadamente 79,000 ejemplos de los datasets `sahil2801/CodeAlpaca-20k` y `TokenBender/code_instructions_122k_alpaca_style`.

## Uso del Modelo desde Hugging Face Hub

El repositorio contiene el modelo final fusionado (merged) y está listo para ser usado directamente para inferencia.

**Repositorio:** [**Ricardouchub/code-specialist-7b**](https://huggingface.co/Ricardouchub/code-specialist-7b)

### Ejemplo de Carga y Uso

Para utilizar este modelo, simplemente cárgalo desde el repositorio del Hub. El siguiente código lo cargará en 4-bits para optimizar el uso de VRAM.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ID del repositorio de tu modelo en Hugging Face
repo_id = "Ricardouchub/code-specialist-7b"

# Configuración de cuantización para cargar en 4-bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Cargar el modelo afinado directamente
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(repo_id)

print("¡Modelo Code Specialist 7B cargado con éxito!")

# Inferencia
pregunta = "Escribe una función en Python que use pandas para agrupar un DataFrame por una columna y calcular la media de otra."
prompt = f"[INST] {pregunta} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Proceso de Desarrollo del Modelo

La creación de este modelo fue un proceso iterativo que implicó la superación de varios desafíos técnicos y metodológicos.

### 1. Definición del Objetivo y Entorno
El objetivo inicial fue realizar un fine-tuning a un modelo de lenguaje de ~7B parámetros que pudiera ejecutarse eficientemente en hardware de consumidor (GPU con 12GB de VRAM). El primer paso crítico fue la configuración de un entorno Conda estable, lo que requirió depurar incompatibilidades entre los drivers de NVIDIA, la versión de CUDA y las dependencias de PyTorch en un sistema operativo Windows.

### 2. Selección y Curación del Dataset
Inicialmente, se intentó un enfoque con un dataset de instrucciones generales (`databricks-dolly-15k`) junto a uno de código (`CodeAlpaca-20k`). Sin embargo, una evaluación cualitativa de los datos filtrados reveló una cantidad significativa de "ruido" o ejemplos no relacionados con programación.

Se tomó la decisión estratégica de pivotar hacia datasets compuestos exclusivamente por instrucciones de código. Se combinaron `CodeAlpaca-20k` y `code_instructions_122k_alpaca_style`, aplicando un filtro estricto para seleccionar únicamente ejemplos de alta calidad de Python y SQL. Este paso fue fundamental para garantizar la especialización del modelo.

### 3. Entrenamiento
El modelo fue entrenado utilizando la técnica de QLoRA, cargando el modelo base en 4-bits para minimizar el consumo de VRAM. Se utilizó la librería `PEFT` para la gestión de los adaptadores LoRA y el `SFTTrainer` de la librería `TRL` para orquestar el proceso de Supervised Fine-Tuning. Durante esta fase, se encontraron y solucionaron incompatibilidades entre las versiones de las librerías `transformers` y `trl`, ajustando los argumentos del entrenador para la versión específica del entorno.

### 4. Evaluación
Se planeó una evaluación cuantitativa utilizando el benchmark estándar `HumanEval`. No obstante, se descubrió que la librería oficial del benchmark no era compatible con Windows. Como alternativa, se implementó un benchmark local con 25 tareas de programación y un sistema de "evaluación humana" para comparar las respuestas del modelo base y del modelo afinado de forma cualitativa, probando la mejora en la calidad, precisión y formato del código generado.

## Instalación y Ejecución de la App Local

Para ejecutar la interfaz de chat de Streamlit en su máquina.

### 1. Requisitos
-   Python 3.10+
-   Conda o venv
-   Una GPU NVIDIA con soporte para CUDA y al menos 12GB de VRAM

### 2. Instalación
Clone el repositorio, cree y active un entorno, e instale las dependencias.

```bash
git clone [https://github.com/Ricardouchub/code-specialist-7b.git](https://github.com/Ricardouchub/code-specialist-7b.git)
cd code-specialist-7b
conda create -n codespecialist python=3.10
conda activate codespecialist
pip install -r requirements.txt
```

### 3. Ejecución

```Bash
streamlit run app.py
```
