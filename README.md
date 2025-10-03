# Code Specialist 7B

**Code Specialist 7B** es un modelo de lenguaje afinado sobre **Mistral-7B-Instruct-v0.3** con un enfoque en generación y explicación de código en **Python** y **SQL**, orientado a tareas de ciencia de datos y análisis.  

El proyecto incluye además una aplicación de chat local construida en **Streamlit** para interactuar con el modelo y el desarrollo del modelo se puede ver en este [Notebook](https://github.com/Ricardouchub/code-specialist-7b/blob/master/code-specialist-7b%20Notebook.ipynb)

---

## Modelo

- **Base:** [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)  
- **Técnica:** Supervised Fine-Tuning (SFT) con adaptadores **LoRA/QLoRA**  
- **Tamaño:** 7B parámetros  
- **Dataset final:** ~79,000 ejemplos filtrados de:  
  - [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)  
  - [TokenBender/code_instructions_122k_alpaca_style](https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style)  

Los ejemplos fueron filtrados y curados para priorizar instrucciones de programación en Python y SQL.

---

## Uso del Modelo

El modelo está disponible en [**Hugging Face Hub: Code-Specialist-7b**](https://huggingface.co/Ricardouchub/code-specialist-7b) y puede cargarse directamente para inferencia:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Repositorio en Hugging Face
model_id = "Ricardouchub/code-specialist-7b"

# Configuración en 4-bit (opcional, para GPUs con VRAM limitada)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Cargar modelo y tokenizador
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Ejemplo de inferencia
prompt = "[INST] Implementa una función en Python para calcular el factorial de un número usando recursividad. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Proceso de Desarrollo

### 1. Objetivo y Entorno
El objetivo fue crear un modelo especializado en generación de código capaz de ejecutarse en hardware de consumo (GPU de 12 GB VRAM).  
Se configuró un entorno estable en Windows con **PyTorch**, **Transformers**, **TRL**, **PEFT** y **BitsAndBytes**, resolviendo incompatibilidades entre versiones y CUDA.

### 2. Dataset
- Se descartaron datasets generales como `databricks-dolly-15k` por exceso de ruido.  
- Se seleccionaron y filtraron datasets de instrucciones de código, garantizando calidad y relevancia.  
- El dataset final se formateó al estilo **Alpaca/Mistral** con la plantilla `[INST] ... [/INST] respuesta`.

### 3. Entrenamiento
- Se aplicó **QLoRA** con el modelo cargado en 4-bit.  
- Se entrenó usando **SFTTrainer** de la librería `trl`.  
- Se ajustaron parámetros de entrenamiento para balancear eficiencia y estabilidad en GPU de 12 GB.

### 4. Evaluación
- No se utilizó `HumanEval` por incompatibilidad en Windows.  
- Se implementó un **benchmark local** con 25 tareas de programación.  
- El modelo fue comparado frente al modelo base para evaluar mejoras en formato, precisión y claridad de código.

---

## Próximos pasos
- Publicar benchmarks completos frente a modelos base y otros especializados.  
- Integración en **Hugging Face Spaces** para un chat interactivo.  
- Extender el fine-tuning a más lenguajes de programación y tareas de análisis de datos.

---

## Autor
**Ricardo Urdaneta**  
[LinkedIn](https://www.linkedin.com/in/ricardourdanetacastro/) | [GitHub](https://github.com/Ricardouchub)  
