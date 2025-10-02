# app.py — Code Specialist 7B (by Ricardo Urdaneta)
import os
import torch
import streamlit as st
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# Evitar banner de bitsandbytes
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

# ======================
# UI & Branding
# ======================
st.set_page_config(page_title="Code Specialist 7B", page_icon="🧠", layout="wide")
st.title("🧠 Code Specialist 7B")
st.caption(
    "Modelo base: **Mistral-7B-Instruct-v0.3** · SFT con subconjuntos de **CodeAlpaca-20k** y **code_instructions_122k (formato Alpaca)** filtrados a Python/SQL · "
    "Entrenado y preparado por **Ricardo Urdaneta**."
)

# Links autor
col_links = st.columns([1,1,6])
with col_links[0]:
    try:
        st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/ricardourdanetacastro/")
    except Exception:
        st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/ricardourdanetacastro/)")
with col_links[1]:
    try:
        st.link_button("💻 GitHub", "https://github.com/Ricardouchub")
    except Exception:
        st.markdown("[💻 GitHub](https://github.com/Ricardouchub)")

st.divider()

# ======================
# Sidebar: configuración
# ======================
st.sidebar.header("⚙️ Configuración del modelo")

default_model_dir = "./Code-Specialist-7b"
model_dir = st.sidebar.text_input("Ruta / repo del modelo", value=default_model_dir)

quant_mode = st.sidebar.selectbox("Cuantización", ["4-bit", "8-bit", "16-bit"], index=0)

# Información de HW
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
st.sidebar.caption(f"CUDA disponible: {torch.cuda.is_available()} | bf16 soportado: {use_bf16}")

# ======================
# Presets
# ======================
st.sidebar.subheader("🎚️ Presets de generación")

# Estado inicial de presets/params
if "gen_params" not in st.session_state:
    st.session_state.gen_params = {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    }

PRESETS = {
    "🔧 Modo Código (determinista)": dict(
        do_sample=False, temperature=0.1, top_p=1.0, repetition_penalty=1.05, max_new_tokens=512
    ),
    "📝 Resumidor (conciso)": dict(
        do_sample=False, temperature=0.2, top_p=0.95, repetition_penalty=1.05, max_new_tokens=1024
    ),
    "🎨 Creativo (brainstorm)": dict(
        do_sample=True, temperature=0.9, top_p=0.95, repetition_penalty=1.0, max_new_tokens=1024
    ),
    "📜 Respuesta Larga": dict(
        do_sample=True, temperature=0.3, top_p=1.0, repetition_penalty=1.02, max_new_tokens=4000
    ),
}

preset_name = st.sidebar.selectbox("Selecciona un preset", list(PRESETS.keys()), index=0)

def apply_preset(name: str):
    st.session_state.gen_params.update(PRESETS[name])

col_p1, col_p2 = st.sidebar.columns([1,1])
if col_p1.button("Aplicar preset"):
    apply_preset(preset_name)
if col_p2.button("Restablecer por defecto"):
    st.session_state.gen_params = {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    }

# Controles ajustables SIEMPRE visibles (puedes retocar tras aplicar preset)
gp = st.session_state.gen_params
do_sample = st.checkbox("do_sample", value=gp["do_sample"])
temperature = st.sidebar.slider("temperature", 0.0, 1.5, float(gp["temperature"]), 0.05)
top_p       = st.sidebar.slider("top_p",       0.05, 1.0, float(gp["top_p"]), 0.05)
repetition_penalty = st.sidebar.slider("repetition_penalty", 1.0, 2.0, float(gp["repetition_penalty"]), 0.01)
max_new     = st.sidebar.slider("max_new_tokens", 16, 4096, int(gp["max_new_tokens"]), 16)

st.sidebar.caption("Nota: Mistral-7B Instruct ~8K de contexto. Entrada + salida ≤ ~8192 tokens.")

system_prompt = st.sidebar.text_area(
    "System prompt (opcional)",
    value="Eres un asistente experto en Python y ciencia de datos. Responde de forma concisa y útil.",
    height=100,
)

# ======================
# Carga perezosa del modelo
# ======================
@st.cache_resource(show_spinner=True)
def load_pipeline(model_id: str, quant: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if quant == "4-bit":
        assert torch.cuda.is_available(), "4-bit requiere GPU."
        bnb4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # Forzar TODO a GPU (4-bit no soporta offload a CPU/disk)
        gpu_id = torch.cuda.current_device()
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb4,
            device_map={"": gpu_id},
            trust_remote_code=True,
        ).eval()
    elif quant == "8-bit":
        bnb8 = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # int8 permite offload a CPU si falta VRAM
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb8,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
    else:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,  # evita warning deprecado torch_dtype
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        ).eval()

    return mdl, tok

try:
    with st.spinner("Cargando modelo..."):
        model, tokenizer = load_pipeline(model_dir, quant_mode)
    st.sidebar.success("Modelo cargado ✅")
except Exception as e:
    st.sidebar.error(f"Error cargando el modelo: {e}")
    st.stop()

# ======================
# Estado del chat
# ======================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# Botones de control
col_a, col_b, col_c = st.columns(3)
if col_a.button("🧹 Limpiar chat"):
    st.session_state.messages = []

if col_b.button("🖴 Exportar historial"):
    st.download_button(
        "Descargar JSON",
        data=str(st.session_state.messages),
        file_name="chat_history.json",
        mime="application/json"
    )

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ======================
# Entrada del usuario
# ======================
user_msg = st.chat_input("Escribe tu mensaje…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Construir mensajes estilo HF
    messages_hf = []
    if system_prompt.strip():
        messages_hf.append({"role": "system", "content": system_prompt.strip()})
    for m in st.session_state.messages:
        # roles compatibles con HF
        role = m["role"] if m["role"] in ("user", "assistant", "system") else "user"
        messages_hf.append({"role": role, "content": m["content"]})

    # Plantilla de chat
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages_hf, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        joined = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages_hf)
        prompt_text = f"[INST] {joined} [/INST]"

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_new),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Generación en streaming
    import threading
    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    ai_resp_container = st.chat_message("assistant")
    ai_text = ""
    with ai_resp_container:
        resp_box = st.empty()
        for token in streamer:
            ai_text += token
            resp_box.markdown(ai_text)

    st.session_state.messages.append({"role": "assistant", "content": ai_text})