# Code Specialist 7B

<p align="left">
  <!-- Status -->
  <img src="https://img.shields.io/badge/Status-Operational-2ECC71?style=flat-square&logo=checkmarx&logoColor=white" alt="Status: Operational"/>

  <!-- Language -->
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11"/>

  <!-- Model -->
  <img src="https://img.shields.io/badge/Mistral-7B--Instruct-FF6F00?style=flat-square&logo=huggingface&logoColor=white" alt="Mistral-7B"/>

  <!-- Technique -->
  <img src="https://img.shields.io/badge/QLoRA-Fine--Tuning-800080?style=flat-square" alt="QLoRA Fine-Tuning"/>

  <!-- Libraries -->
  <img src="https://img.shields.io/badge/Transformers-4.56.2-FFAE1A?style=flat-square&logo=huggingface&logoColor=white" alt="Transformers"/>
  <!-- App -->
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>

  <!-- Hugging Face -->
  <a href="https://huggingface.co/Ricardouchub/code-specialist-7b">
    <img src="https://img.shields.io/badge/HuggingFace-Repo-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace Repo"/>
  </a>
</p>


[**Code Specialist 7B**](https://huggingface.co/Ricardouchub/code-specialist-7b) is a language model fine-tuned on **Mistral-7B-Instruct-v0.3** with a focus on code generation and explanation in **Python** and **SQL**, aimed at data science and analytics tasks.  

The project also includes a local chat application built with **Streamlit** to interact with the model, and the model development process is documented in this [Notebook](https://github.com/Ricardouchub/code-specialist-7b/blob/master/code-specialist-7b%20Notebook.ipynb).

---

## Model

- **Base:** [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)  
- **Technique:** SFT with *QLoRA* (Quantized Low-Rank Adaptation).  
- **Size:** 7B parameters  
- **Final dataset:** ~79,000 filtered examples from:  
  - [sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)  
  - [TokenBender/code_instructions_122k_alpaca_style](https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style)  

The examples were filtered and curated to prioritize programming instructions in Python and SQL.

## Training

| **Aspect**        | **Detail** |
|--------------------|-------------|
| **Method**         | SFT with QLoRA |
| **Frameworks**     | `transformers`, `trl`, `peft`, `bitsandbytes` |
| **Hardware**       | GPU 12 GB VRAM (4-bit quantization for training) |

## Main Hyperparameters

| **Parameter** | **Value** |
|----------------|-----------|
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 4 |
| `learning_rate` | 0.0002 |
| `num_train_epochs` | 1 |
| `max_seq_length` | 1024 |

---

## Development Process

### 1. Objective and Environment
The goal was to create a code-oriented generative model that can run on consumer hardware (GPUs with 12 GB VRAM).  
A stable Windows environment was configured with **PyTorch**, **Transformers**, **TRL**, **PEFT**, and **BitsAndBytes**, resolving version and CUDA incompatibilities.

### 2. Dataset
- General-purpose datasets such as `databricks-dolly-15k` were discarded due to excessive noise.  
- Code instruction datasets were selected and filtered to ensure quality and relevance.  
- The final dataset was formatted in the **Alpaca/Mistral** style using the `[INST] ... [/INST] response` template.

### 3. Training
- **QLoRA** was applied with the model loaded in 4-bit.  
- Training was performed with the `SFTTrainer` from the `trl` library.  
- Training hyperparameters were tuned to balance efficiency and stability on a 12 GB GPU.

### 4. Evaluation
- `HumanEval` was not used due to incompatibility on Windows.  
- A **local benchmark** with 25 programming tasks was implemented.  
- The fine-tuned model was compared against the base model to evaluate improvements in formatting, accuracy, and code clarity.

--- 

## Using the Model from Hugging Face

The model is available on [**Hugging Face Hub: Code-Specialist-7b**](https://huggingface.co/Ricardouchub/code-specialist-7b) and can be loaded directly for inference:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# HF Login Token
login(token="HF_TOKEN_HERE")

# Hugging Face repository
model_id = "Ricardouchub/code-specialist-7b"

# 4-bit configuration (optional, for GPUs with limited VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inference example
prompt = "[INST] Implement a Python function that computes the factorial of a number using recursion. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Environment Reproduction

### Hardware requirements
- NVIDIA GPU with 12 GB of VRAM or more for 4-bit quantization (tested on RTX 3060).  
- GPUs with 8 GB can run the model in 8-bit or 16-bit with a shorter context window.  

Install [uv](https://docs.astral.sh/uv/)

Install PyTorch for your hardware:
- CUDA 12.1 GPU:
  ```powershell
  uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
  ```

Then sync the rest of the dependencies in `pyproject.toml`:
```powershell
uv sync
```

`uv sync` creates an isolated virtual environment (defaults to `.venv`) and installs `transformers`, `accelerate`, `streamlit`, `peft`, `trl`, `sentencepiece`, `safetensors`, plus conditional `bitsandbytes`.

### Downloading the model weights
1. Clone from GitHub: `git clone https://github.com/Ricardouchub/code-specialist-7b.git`.  
2. To use only the LoRA adapters: locate the `lora-adapters` folder and merge it with the base model `mistralai/Mistral-7B-Instruct-v0.3`.

### Launching the Streamlit application
```powershell
uv run streamlit run app.py
```

- Set the model path in the sidebar if it differs from the default value.  
- Choose the quantization mode (4-bit/8-bit/16-bit) according to the available VRAM.  
- Customize the `system prompt`, presets, or generation parameters before sending messages.

---

## Author
**Ricardo Urdaneta**  
[LinkedIn](https://www.linkedin.com/in/ricardourdanetacastro/) | [GitHub](https://github.com/Ricardouchub)  
