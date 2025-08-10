from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
import torch
from threading import Thread
import gradio as gr
import spaces
import re
import logging
import os
from peft import PeftModel

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

def format_conversation_history(chat_history):
    messages = []
    for item in chat_history:
        role = item["role"]
        content = item["content"]
        if isinstance(content, list):
            content = content[0]["text"] if content and "text" in content[0] else str(content)
        messages.append({"role": role, "content": content})
    return messages

def format_analysis_response(text):
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
def generate_response(input_data, chat_history, max_new_tokens, model_identity, system_prompt, developer_prompt, reasoning_effort, temperature, top_p, top_k, repetition_penalty):
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

demo = gr.ChatInterface(
    fn=generate_response,
    additional_inputs=[
        gr.Slider(label="Max new tokens", minimum=64, maximum=4096, step=1, value=2048),
        gr.Textbox(
            label="Model Identity",
            value=MODEL_IDENTITY,
            lines=3,
            placeholder="Optional identity/persona for the model"
        ),
        gr.Textbox(
            label="System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            lines=4,
            placeholder="Change system prompt"
        ),
        gr.Textbox(
            label="Developer Prompt",
            value=DEFAULT_DEVELOPER_PROMPT,
            lines=4,
            placeholder="Optional developer instructions"
        ),
        gr.Dropdown(
            label="Reasoning Effort",
            choices=["low", "medium", "high"],
            value=DEFAULT_REASONING_EFFORT,
            interactive=True,
        ),
        gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=0.7),
        gr.Slider(label="Top-p", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=100, step=1, value=50),
        gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.0)
    ],
    examples=[
        [{"text": "Explain Newton's laws clearly and concisely with mathematical formulas"}],
        [{"text": "Write a Python function to calculate the Fibonacci sequence"}],
        [{"text": "What are the benefits of open weight AI models? Include analysis."}],
        [{"text": "Solve this equation: $x^2 + 5x + 6 = 0$"}],
    ],
    cache_examples=False,
    type="messages",
    description=f"""

# 🙋🏻‍♂️Welcome to 🌟{MODEL_NAME} Demo !

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
    """,
    fill_height=True,
    textbox=gr.Textbox(
        label="Query Input",
        placeholder="Type your prompt (supports LaTeX: $x^2 + y^2 = z^2$)"
    ),
    stop_btn="Stop Generation",
    multimodal=False,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)