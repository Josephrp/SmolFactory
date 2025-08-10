from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
import torch
from threading import Thread
import gradio as gr
import spaces
import re
import logging
import os
import json
from peft import PeftModel
from typing import Any, Dict, List, Generator

# ----------------------------------------------------------------------
# Environment Variables Configuration
# ----------------------------------------------------------------------

# Get model configuration from environment variables
BASE_MODEL_ID = os.getenv('BASE_MODEL_ID', 'openai/gpt-oss-20b')
LORA_MODEL_ID = os.getenv('LORA_MODEL_ID', os.getenv('HF_MODEL_ID', 'Tonic/gpt-oss-20b-multilingual-reasoner'))
MODEL_NAME = os.getenv('MODEL_NAME', 'GPT-OSS Multilingual Reasoner')
MODEL_SUBFOLDER = os.getenv('MODEL_SUBFOLDER', '')

# Optional persona and prompts derived from training config
MODEL_IDENTITY = os.getenv('MODEL_IDENTITY', '')
DEFAULT_SYSTEM_PROMPT = os.getenv('SYSTEM_MESSAGE', MODEL_IDENTITY or 'You are a helpful assistant. Reasoning: medium')
DEFAULT_DEVELOPER_PROMPT = os.getenv('DEVELOPER_MESSAGE', '')
DEFAULT_REASONING_EFFORT = os.getenv('REASONING_EFFORT', 'medium')

# If the LORA_MODEL_ID is the same as BASE_MODEL_ID, this is a merged model, not LoRA
USE_LORA = LORA_MODEL_ID != BASE_MODEL_ID and not LORA_MODEL_ID.startswith(BASE_MODEL_ID)

print(f"🔧 Configuration:")
print(f"   Base Model: {BASE_MODEL_ID}")
print(f"   Model ID: {LORA_MODEL_ID}")
print(f"   Model Name: {MODEL_NAME}")
print(f"   Model Subfolder: {MODEL_SUBFOLDER}")
print(f"   Use LoRA: {USE_LORA}")

# ----------------------------------------------------------------------
# Branding/Owner customization (override via Space variables)
# ----------------------------------------------------------------------

HF_USERNAME = os.getenv("HF_USERNAME", "")
BRAND_OWNER_NAME = os.getenv("BRAND_OWNER_NAME", HF_USERNAME or "Tonic")
BRAND_TEAM_NAME = os.getenv("BRAND_TEAM_NAME", f"Team{BRAND_OWNER_NAME}" if BRAND_OWNER_NAME else "TeamTonic")

BRAND_DISCORD_URL = os.getenv("BRAND_DISCORD_URL", "https://discord.gg/qdfnvSPcqP")

# Hugging Face org/links
_default_hf_org = os.getenv("HF_ORG", HF_USERNAME) or "MultiTransformer"
BRAND_HF_ORG = os.getenv("BRAND_HF_ORG", _default_hf_org)
BRAND_HF_LABEL = os.getenv("BRAND_HF_LABEL", BRAND_HF_ORG)
BRAND_HF_URL = os.getenv("BRAND_HF_URL", f"https://huggingface.co/{BRAND_HF_ORG}")

# GitHub org/links
_default_gh_org = os.getenv("GITHUB_ORG", "tonic-ai")
BRAND_GH_ORG = os.getenv("BRAND_GH_ORG", _default_gh_org)
BRAND_GH_LABEL = os.getenv("BRAND_GH_LABEL", BRAND_GH_ORG)
BRAND_GH_URL = os.getenv("BRAND_GH_URL", f"https://github.com/{BRAND_GH_ORG}")

# Project link (optional)
BRAND_PROJECT_NAME = os.getenv("BRAND_PROJECT_NAME", "MultiTonic")
BRAND_PROJECT_URL = os.getenv("BRAND_PROJECT_URL", "https://github.com/MultiTonic")

# ----------------------------------------------------------------------
# Title/Description content (Markdown/HTML)
# ----------------------------------------------------------------------

TITLE_MD = f"# 🙋🏻‍♂️ Welcome to 🌟{BRAND_OWNER_NAME}'s ⚕️{MODEL_NAME} Demo !"

DESCRIPTION_MD = f"""
**Model**: `{LORA_MODEL_ID}`  
**Base**: `{BASE_MODEL_ID}`

✨ **Enhanced Features:**
- 🧠 **Advanced Reasoning**: Detailed analysis and step-by-step thinking
- 📊 **LaTeX Support**: Mathematical formulas rendered beautifully (use `$` or `$$`)
- 🎯 **Improved Formatting**: Clear separation of reasoning and final responses
- 📝 **Smart Logging**: Better error handling and request tracking

💡 **Usage Tips:**
- Adjust reasoning level in system prompt (e.g., "Reasoning: high")
- Use LaTeX for math: `$E = mc^2$` or `$$\\int x^2 dx$$`
- Wait a couple of seconds initially for model loading
"""

# ----------------------------------------------------------------------
# Examples configuration with robust fallbacks
# ----------------------------------------------------------------------

def _normalize_examples(string_items: List[str]) -> List[List[Dict[str, str]]]:
    """Convert a list of strings to Gradio ChatInterface examples format."""
    return [[{"text": s}] for s in string_items]

DEFAULT_EXAMPLES_GENERAL: List[str] = [
    "Explain Newton's laws clearly and concisely with mathematical formulas",
    "Write a Python function to calculate the Fibonacci sequence",
    "What are the benefits of open weight AI models? Include analysis.",
    "Solve this equation: $x^2 + 5x + 6 = 0$",
]

DEFAULT_EXAMPLES_MEDICAL: List[str] = [
    "A 68-year-old man complains of several blisters arising over the back and trunk for the preceding 2 weeks. He takes no medications and has not noted systemic symptoms such as fever, sore throat, weight loss, or fatigue. The general physical examination is normal. The oral mucosa and the lips are normal. Several 2- to 3-cm bullae are present over the trunk and back. A few excoriations where the blisters have ruptured are present. The remainder of the skin is normal, without erythema or scale. What is the best diagnostic approach at this time?",
    "A 28-year-old woman, gravida 2, para 1, at 40 weeks of gestation is admitted to the hospital in active labor. The patient has attended many prenatal appointments and followed her physician's advice about screening for diseases, laboratory testing, diet, and exercise. Her pregnancy has been uncomplicated. She has no history of a serious illness. Her first child was delivered via normal vaginal delivery. Her vital signs are within normal limits. Cervical examination shows 100% effacement and 10 cm dilation. A cardiotocograph is shown. Which of the following is the most appropriate initial step in management?",
    "An 18-year-old woman has eaten homemade preserves. Eighteen hours later, she develops diplopia, dysarthria, and dysphagia. She presents to the emergency room for assessment and on examination her blood pressure is 112/74 mmHg, heart rate 110/min, and respirations 20/min. The pertinent findings are abnormal extraocular movements due to cranial nerve palsies, difficulty swallowing and a change in her voice. The strength in her arms is 4/5 and 5/5 in her legs, and the reflexes are normal. Which of the following is the most likely causative organism?",
    "What are you & who made you?",
]

_examples_type_env = os.getenv("EXAMPLES_TYPE", "medical").strip().lower()
_disable_examples = os.getenv("DISABLE_EXAMPLES", "false").strip().lower() in {"1", "true", "yes"}
_examples_json_raw = os.getenv("EXAMPLES_JSON") or os.getenv("CUSTOM_EXAMPLES_JSON")

EXAMPLES_FINAL: List[List[Dict[str, str]]]
if _disable_examples:
    EXAMPLES_FINAL = []
else:
    custom_items: List[str] = []
    if _examples_json_raw:
        try:
            parsed = json.loads(_examples_json_raw)
            if isinstance(parsed, list) and parsed:
                # Accept list[str]
                if all(isinstance(x, str) for x in parsed):
                    custom_items = parsed  # type: ignore[assignment]
                # Accept list[dict{"text": str}]
                elif all(isinstance(x, dict) and "text" in x and isinstance(x["text"], str) for x in parsed):
                    custom_items = [x["text"] for x in parsed]
        except Exception:
            custom_items = []

    if custom_items:
        EXAMPLES_FINAL = _normalize_examples(custom_items)
    else:
        if _examples_type_env in {"med", "medical", "health"}:
            EXAMPLES_FINAL = _normalize_examples(DEFAULT_EXAMPLES_MEDICAL)
        else:
            EXAMPLES_FINAL = _normalize_examples(DEFAULT_EXAMPLES_GENERAL)

JOIN_US_MD = f"""
## Join us :
🌟{BRAND_TEAM_NAME}🌟 is always making cool demos! Join our active builder's 🛠️community 👻
[Join us on Discord]({BRAND_DISCORD_URL})
On 🤗Hugging Face: [{BRAND_HF_LABEL}]({BRAND_HF_URL})
On 🌐GitHub: [{BRAND_GH_LABEL}]({BRAND_GH_URL}) & contribute to 🌟 [{BRAND_PROJECT_NAME}]({BRAND_PROJECT_URL})
🤗 Big thanks to the Hugging Face team for the community support 🤗
"""

# ----------------------------------------------------------------------
# KaTeX delimiter config for Gradio
# ----------------------------------------------------------------------

LATEX_DELIMS = [
    {"left": "$$",  "right": "$$",  "display": True},
    {"left": "$",   "right": "$",   "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model
try:
    if USE_LORA:
        # Load base model and LoRA adapter separately
        print(f"🔄 Loading base model: {BASE_MODEL_ID}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="kernels-community/vllm-flash-attn3"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # Load the LoRA adapter
        try:
            print(f"🔄 Loading LoRA adapter: {LORA_MODEL_ID}")
            if MODEL_SUBFOLDER and MODEL_SUBFOLDER.strip():
                model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID, subfolder=MODEL_SUBFOLDER)
            else:
                model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID)
            print("✅ LoRA model loaded successfully!")
        except Exception as lora_error:
            print(f"⚠️ LoRA adapter failed to load: {lora_error}")
            print("🔄 Falling back to base model...")
            model = base_model
    else:
        # Load merged/fine-tuned model directly
        print(f"🔄 Loading merged model: {LORA_MODEL_ID}")
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "attn_implementation": "kernels-community/vllm-flash-attn3"
        }
        
        if MODEL_SUBFOLDER and MODEL_SUBFOLDER.strip():
            model = AutoModelForCausalLM.from_pretrained(LORA_MODEL_ID, subfolder=MODEL_SUBFOLDER, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_ID, subfolder=MODEL_SUBFOLDER)
        else:
            model = AutoModelForCausalLM.from_pretrained(LORA_MODEL_ID, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_ID)
        print("✅ Merged model loaded successfully!")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

def format_conversation_history(chat_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize Gradio chat history items into a list of role/content messages."""
    messages: List[Dict[str, str]] = []
    for item in chat_history:
        role = item["role"]
        content = item["content"]
        if isinstance(content, list):
            content = content[0]["text"] if content and "text" in content[0] else str(content)
        messages.append({"role": role, "content": content})
    return messages

def format_analysis_response(text: str) -> str:
    """Enhanced response formatting with better structure and LaTeX support."""
    # Look for analysis section followed by final response
    m = re.search(r"analysis(.*?)assistantfinal", text, re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()
        response = text.split("assistantfinal", 1)[-1].strip()
        
        # Clean up the reasoning section
        reasoning = re.sub(r'^analysis\s*', '', reasoning, flags=re.IGNORECASE).strip()
        
        # Format with improved structure
        formatted = (
            f"**🤔 Analysis & Reasoning:**\n\n"
            f"*{reasoning}*\n\n"
            f"---\n\n"
            f"**💬 Final Response:**\n\n{response}"
        )
        
        # Ensure LaTeX delimiters are balanced
        if formatted.count("$") % 2:
            formatted += "$"
            
        return formatted
    
    # Fallback: clean up the text and return as-is
    cleaned = re.sub(r'^analysis\s*', '', text, flags=re.IGNORECASE).strip()
    if cleaned.count("$") % 2:
        cleaned += "$"
    return cleaned

@spaces.GPU(duration=60)
def generate_response(
    input_data: str,
    chat_history: List[Dict[str, Any]],
    max_new_tokens: int,
    model_identity: str,
    system_prompt: str,
    developer_prompt: str,
    reasoning_effort: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Generator[str, None, None]:
    """Stream tokens as they are generated, yielding formatted partial/final outputs."""
    if not input_data.strip():
        yield "Please enter a prompt."
        return
        
    # Log the request
    logging.info(f"[User] {input_data}")
    logging.info(f"[System] {system_prompt} | Temp={temperature} | Max tokens={max_new_tokens}")
    
    new_message = {"role": "user", "content": input_data}
    # Combine model identity with system prompt for a single system message
    combined_parts = []
    if model_identity and model_identity.strip():
        combined_parts.append(model_identity.strip())
    if system_prompt and system_prompt.strip():
        combined_parts.append(system_prompt.strip())
    if reasoning_effort and isinstance(reasoning_effort, str) and reasoning_effort.strip():
        # Append explicit reasoning directive
        combined_parts.append(f"Reasoning: {reasoning_effort.strip()}")
    combined_system = "\n\n".join(combined_parts).strip()
    system_message = ([{"role": "system", "content": combined_system}] if combined_system else [])
    developer_message = [{"role": "developer", "content": developer_prompt}] if developer_prompt else []
    processed_history = format_conversation_history(chat_history)
    messages = system_message + developer_message + processed_history + [new_message]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback: merge developer prompt into system prompt if template doesn't support 'developer' role
        fallback_sys = combined_system
        if developer_prompt:
            fallback_sys = (fallback_sys + ("\n\n[Developer]\n" if fallback_sys else "[Developer]\n") + developer_prompt).strip()
        fallback_messages = ([{"role": "system", "content": fallback_sys}] if fallback_sys else []) + processed_history + [new_message]
        prompt = tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Create streamer for proper streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Prepare generation kwargs
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
        "use_cache": True
    }
    
    # Tokenize input using the chat template
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs={**inputs, **generation_kwargs})
    thread.start()
    
    # Stream the response with enhanced formatting
    collected_text = ""
    buffer = ""
    yielded_once = False
    
    try:
        for chunk in streamer:
            if not chunk:
                continue
                
            collected_text += chunk
            buffer += chunk
            
            # Initial yield to show immediate response
            if not yielded_once:
                yield chunk
                buffer = ""
                yielded_once = True
                continue
            
            # Yield accumulated text periodically for smooth streaming
            if "\n" in buffer or len(buffer) > 150:
                # Use enhanced formatting for partial text
                partial_formatted = format_analysis_response(collected_text)
                yield partial_formatted
                buffer = ""
        
        # Final formatting with complete text
        final_formatted = format_analysis_response(collected_text)
        yield final_formatted
        
    except Exception as e:
        logging.exception("Generation streaming failed")
        yield f"❌ Error during generation: {e}"

# ----------------------------------------------------------------------
# UI/Styling: CSS + custom Chatbot + two-column description
# ----------------------------------------------------------------------

APP_CSS = """
#main_chatbot {height: calc(100vh - 120px);} /* Increase chatbot viewport height */
.gradio-container {min-height: 100vh;}
"""

description_html = f"""
<div style=\"display:flex; gap: 16px; align-items:flex-start; flex-wrap: wrap\">
  <div style=\"flex: 1 1 60%; min-width: 300px;\">
  {DESCRIPTION_MD}
  </div>
  <div style=\"flex: 1 1 35%; min-width: 260px;\">
  {JOIN_US_MD}
  </div>
  </div>
"""

custom_chatbot = gr.Chatbot(label="Chatbot", elem_id="main_chatbot", latex_delimiters=LATEX_DELIMS)

demo = gr.ChatInterface(
        fn=generate_response,
        chatbot=custom_chatbot,
        title=f"🙋🏻‍♂️ Welcome to 🌟{BRAND_OWNER_NAME}'s ⚕️{MODEL_NAME} Demo !",
        description=description_html,
        additional_inputs=[
            gr.Slider(label="Max new tokens", minimum=64, maximum=4096, step=1, value=2048),
            gr.Textbox(
                label="🪪Model Identity",
                value=MODEL_IDENTITY,
                lines=1,
                placeholder="Optional identity/persona for the model"
            ),
            gr.Textbox(
                label="🤖System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=1,
                placeholder="Change system prompt"
            ),
            gr.Textbox(
                label="👨🏻‍💻Developer Prompt",
                value=DEFAULT_DEVELOPER_PROMPT,
                lines=1,
                placeholder="Optional developer instructions"
            ),
            gr.Dropdown(
                label="🧠Reasoning Effort",
                choices=["low", "medium", "high"],
                value=DEFAULT_REASONING_EFFORT,
                interactive=True,
            ),
            gr.Slider(label="🌡️Temperature", minimum=0.1, maximum=2.0, step=0.1, value=0.7),
            gr.Slider(label="↗️Top-p", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
            gr.Slider(label="🔝Top-k", minimum=1, maximum=100, step=1, value=50),
            gr.Slider(label="🦜Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.0),
            ],
        additional_inputs_accordion=gr.Accordion(label="🔧Advanced Inputs", open=False),
        examples=EXAMPLES_FINAL,
        cache_examples=False,
        type="messages",
        fill_height=True,
        fill_width=True,
        textbox=gr.Textbox(
            label="Query Input",
            placeholder="Type your prompt (supports LaTeX: $x^2 + y^2 = z^2$)"
        ),
        stop_btn="Stop Generation",
        multimodal=False,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True, share=True)