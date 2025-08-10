import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
from typing import List, Dict, Any, Optional
import logging
import spaces
import os
import sys
import requests

# Set torch to use float32 for better compatibility with quantized models
torch.set_default_dtype(torch.float32)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Model & UI configuration (env-driven)
# ----------------------------------------------------------------------

# Get model ID from environment variable or use default
MAIN_MODEL_ID = os.getenv("HF_MODEL_ID", "Tonic/petite-elle-L-aime-3-sft")
MODEL_SUBFOLDER = os.getenv("MODEL_SUBFOLDER", "int4")  # Default to int4 for CPU deployment
MODEL_NAME = os.getenv("MODEL_NAME", "SmolLM3 Fine-tuned Model")

# Branding/Owner customization
HF_USERNAME = os.getenv("HF_USERNAME", "")
BRAND_OWNER_NAME = os.getenv("BRAND_OWNER_NAME", HF_USERNAME or "Tonic")
BRAND_TEAM_NAME = os.getenv("BRAND_TEAM_NAME", f"Team{BRAND_OWNER_NAME}" if BRAND_OWNER_NAME else "TeamTonic")

BRAND_DISCORD_URL = os.getenv("BRAND_DISCORD_URL", "https://discord.gg/qdfnvSPcqP")

_default_hf_org = os.getenv("BRAND_HF_ORG") or HF_USERNAME or "MultiTransformer"
BRAND_HF_ORG = _default_hf_org
BRAND_HF_LABEL = os.getenv("BRAND_HF_LABEL", BRAND_HF_ORG)
BRAND_HF_URL = os.getenv("BRAND_HF_URL", f"https://huggingface.co/{BRAND_HF_ORG}")

_default_gh_org = os.getenv("BRAND_GH_ORG") or HF_USERNAME or "tonic-ai"
BRAND_GH_ORG = _default_gh_org
BRAND_GH_LABEL = os.getenv("BRAND_GH_LABEL", BRAND_GH_ORG)
BRAND_GH_URL = os.getenv("BRAND_GH_URL", f"https://github.com/{BRAND_GH_ORG}")

BRAND_PROJECT_NAME = os.getenv("BRAND_PROJECT_NAME", "MultiTonic")
BRAND_PROJECT_URL = os.getenv("BRAND_PROJECT_URL", "https://github.com/MultiTonic")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_MESSAGE",
    "Tu es un assistant francophone rigoureux et bienveillant."
)
title = f"# 🙋🏻‍♂️ Bienvenue chez 🌟{BRAND_OWNER_NAME} — {MODEL_NAME}"
description = f"A fine-tuned version of SmolLM3-3B optimized for conversations. This is the {MODEL_SUBFOLDER} quantized version for efficient deployment."
presentation1 = """
### 🎯 Features
- **Multilingual Support**: English, French, Italian, Portuguese, Chinese, Arabic
- **Quantized Model**: Optimized for deployment with memory reduction
- **Interactive Chat Interface**: Real-time conversation with the model
- **Customizable System Prompt**: Define the assistant's personality and behavior
- **Thinking Mode**: Enable reasoning mode with thinking tags
"""
presentation2 = """### 🎯 Fonctionnalités
* **Support multilingue** : Anglais, Français, Italien, Portugais, Chinois, Arabe
* **Modèle quantifié** : Optimisé pour un déploiement avec réduction de mémoire
* **Interface de chat interactive** : Conversation en temps réel avec le modèle
* **Invite système personnalisable** : Définissez la personnalité et le comportement de l'assistant
* **Mode Réflexion** : Activez le mode raisonnement avec des balises de réflexion
"""
joinus = """
## Join us :
🌟TeamTonic🌟 is always making cool demos! Join our active builder's 🛠️community 👻 [![Join us on Discord](https://img.shields.io/discord/1109943800132010065?label=Discord&logo=discord&style=flat-square)](https://discord.gg/qdfnvSPcqP) On 🤗Huggingface:[MultiTransformer](https://huggingface.co/MultiTransformer) On 🌐Github: [Tonic-AI](https://github.com/tonic-ai) & contribute to🌟 [Build Tonic](https://git.tonic-ai.com/contribute)🤗Big thanks to Yuvi Sharma and all the folks at huggingface for the community grant 🤗
"""


def download_chat_template():
    """Download the chat template from the main repository"""
    try:
        chat_template_url = f"https://huggingface.co/{MAIN_MODEL_ID}/raw/main/chat_template.jinja"
        logger.info(f"Downloading chat template from {chat_template_url}")
        
        response = requests.get(chat_template_url, timeout=30)
        response.raise_for_status()
        
        chat_template_content = response.text
        logger.info("Chat template downloaded successfully")
        return chat_template_content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading chat template: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading chat template: {e}")
        return None


# ----------------------------------------------------------------------
# Examples configuration with robust fallbacks
# ----------------------------------------------------------------------

def _normalize_examples_flat(string_items: List[str]) -> List[str]:
    """Convert list[str] to a flat list usable by gr.Examples targeting a single input."""
    return list(string_items)

DEFAULT_EXAMPLES_GENERAL: List[str] = [
    "Explique les lois de Newton avec des formules mathématiques",
    "Écris une fonction Python pour calculer la suite de Fibonacci",
    "Quels sont les avantages des modèles IA open-weight ?",
    "Résous cette équation : $x^2 + 5x + 6 = 0$",
]

DEFAULT_EXAMPLES_MEDICAL: List[str] = [
    "Un homme de 68 ans présente plusieurs bulles sur le dos et le tronc depuis 2 semaines. Quel est le meilleur diagnostic à ce stade ?",
    "Une femme de 28 ans en travail à 40 semaines de gestation : quelle est la première étape de prise en charge ?",
    "Une femme de 18 ans a consommé des conserves maison et présente diplopie et dysphagie. Quel est l'agent causal le plus probable ?",
]

_examples_type_env = (os.getenv("EXAMPLES_TYPE", "general") or "general").strip().lower()
_disable_examples = os.getenv("DISABLE_EXAMPLES", "false").strip().lower() in {"1", "true", "yes"}
_examples_json_raw = os.getenv("EXAMPLES_JSON") or os.getenv("CUSTOM_EXAMPLES_JSON")

if not _disable_examples and _examples_json_raw:
    try:
        parsed = json.loads(_examples_json_raw)
        if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
            EXAMPLES_FLAT: List[str] = _normalize_examples_flat(parsed)
        else:
            EXAMPLES_FLAT = []
    except Exception:
        EXAMPLES_FLAT = []
else:
    if _examples_type_env in {"med", "medical", "santé", "sante"}:
        EXAMPLES_FLAT = _normalize_examples_flat(DEFAULT_EXAMPLES_MEDICAL)
    else:
        EXAMPLES_FLAT = _normalize_examples_flat(DEFAULT_EXAMPLES_GENERAL)

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info(f"Loading tokenizer from {MAIN_MODEL_ID}")
        if MODEL_SUBFOLDER and MODEL_SUBFOLDER.strip():
            tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_ID, subfolder=MODEL_SUBFOLDER)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_ID)
        chat_template = download_chat_template()
        if chat_template:
            tokenizer.chat_template = chat_template
            logger.info("Chat template downloaded and set successfully")
        else:
            logger.warning("Could not download chat template, using default")

        logger.info(f"Loading model from {MAIN_MODEL_ID}")
        model_kwargs = {
            "device_map": "auto" if DEVICE == "cuda" else "cpu",
            "torch_dtype": torch.bfloat16, 
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        logger.info(f"Model loading parameters: {model_kwargs}")
        if MODEL_SUBFOLDER and MODEL_SUBFOLDER.strip():
            model = AutoModelForCausalLM.from_pretrained(MAIN_MODEL_ID, subfolder=MODEL_SUBFOLDER, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(MAIN_MODEL_ID, **model_kwargs)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(f"Model config: {model.config if model else 'Model not loaded'}")
        return False


def create_prompt(system_message, user_message, enable_thinking=True):
    """Create prompt using the model's chat template"""
    try:
        formatted_messages = []
        if system_message and system_message.strip():
            formatted_messages.append({"role": "system", "content": system_message})
        formatted_messages.append({"role": "user", "content": user_message})        
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )        
        if not enable_thinking:
            prompt += " /no_think"
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        return ""

@spaces.GPU(duration=94)
def generate_response(message, history, system_message, max_tokens, temperature, top_p, do_sample, enable_thinking=True):
    """Generate response using the model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please wait for the model to load."
    full_prompt = create_prompt(system_message, message, enable_thinking)
     
    if not full_prompt:
        return "Error: Failed to create prompt."
        
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)        
    logger.info(f"Input tensor shapes: {[(k, v.shape, v.dtype) for k, v in inputs.items()]}")

    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)        
        assistant_response = response[len(full_prompt):].strip()
        assistant_response = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', assistant_response, flags=re.DOTALL)
        if not enable_thinking:
            assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
        
        assistant_response = assistant_response.strip()
        
        return assistant_response

def user(user_message, history):
    """Add user message to history"""
    if history is None:
        history = []
    return "", history + [{"role": "user", "content": user_message}]

def bot(history, system_prompt, max_length, temperature, top_p, advanced_checkbox, enable_thinking):
    """Generate bot response"""
    if not history:
        return history    
    user_message = history[-1]["content"] if history else ""
    
    do_sample = advanced_checkbox
    bot_message = generate_response(user_message, history, system_prompt, max_length, temperature, top_p, do_sample, enable_thinking)
    history.append({"role": "assistant", "content": bot_message})
    return history

# Load model on startup
logger.info("Starting model loading process...")
load_model()

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Row(): 
        gr.Markdown(title)
    with gr.Row():
        gr.Markdown(description)
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown(presentation1)
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown(presentation2)
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown(joinus)
        with gr.Column(scale=1):
            pass  # Empty column for balance
    
    with gr.Row():
        with gr.Column(scale=2):
            system_prompt = gr.TextArea(
                label="📑 Contexte", 
                placeholder="Tu es TonicIA, un assistant francophone rigoureux et bienveillant.", 
                lines=5,
                value=DEFAULT_SYSTEM_PROMPT
            )
            user_input = gr.TextArea(
                label="🤷🏻‍♂️ Message", 
                placeholder="Bonjour je m'appel Tonic!", 
                lines=2
            )
            if EXAMPLES_FLAT:
                gr.Examples(label="Exemples", examples=EXAMPLES_FLAT, inputs=user_input)
            advanced_checkbox = gr.Checkbox(label="🧪 Advanced Settings", value=False)
            with gr.Column(visible=False) as advanced_settings:
                max_length = gr.Slider(
                    label="📏 Longueur de la réponse", 
                    minimum=10, 
                    maximum=556, 
                    value=120, 
                    step=1
                )
                temperature = gr.Slider(
                    label="🌡️ Température", 
                    minimum=0.01, 
                    maximum=1.0, 
                    value=0.5, 
                    step=0.01
                )
                top_p = gr.Slider(
                    label="⚛️ Top-p (Echantillonnage)", 
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.95, 
                    step=0.01
                )
                enable_thinking = gr.Checkbox(label="Mode Réflexion", value=True)
            
            generate_button = gr.Button(value=f"🤖 {MODEL_NAME}")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label=f"🤖 {MODEL_NAME}", type="messages", value=[])
    
    generate_button.click(
        user,
        [user_input, chatbot],
        [user_input, chatbot],
        queue=False
    ).then(
        bot,
        [chatbot, system_prompt, max_length, temperature, top_p, advanced_checkbox, enable_thinking],
        chatbot
    )

    advanced_checkbox.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[advanced_checkbox],
        outputs=[advanced_settings]
    )

if __name__ == "__main__":

    demo.queue()
    demo.launch(ssr_mode=False, mcp_server=True)