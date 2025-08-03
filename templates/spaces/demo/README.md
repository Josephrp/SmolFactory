---
title: Petite LLM 3
emoji: 💃🏻
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
license: mit
short_description: Smollm3 for French Understanding
---

# 🤖 Petite Elle L'Aime 3 - Chat Interface

A complete Gradio application for the [Petite Elle L'Aime 3](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft) model, featuring the int4 quantized version for efficient CPU deployment.

## 🚀 Features

- **Multilingual Support**: English, French, Italian, Portuguese, Chinese, Arabic
- **Int4 Quantization**: Optimized for CPU deployment with ~50% memory reduction
- **Interactive Chat Interface**: Real-time conversation with the model
- **Customizable System Prompt**: Define the assistant's personality and behavior
- **Thinking Mode**: Enable reasoning mode with thinking tags
- **Responsive Design**: Modern UI following the reference layout
- **Chat Template Integration**: Proper Jinja template formatting
- **Automatic Model Download**: Downloads int4 model at build time

## 📋 Model Information

- **Base Model**: SmolLM3-3B
- **Parameters**: ~3B
- **Context Length**: 128k
- **Quantization**: int4 (CPU optimized)
- **Memory Reduction**: ~50%
- **Languages**: English, French, Italian, Portuguese, Chinese, Arabic

## 🛠️ Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Petite-LLM-3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Local Development

Run the application locally:
```bash
python app.py
```

The application will be available at `http://localhost:7860`

### Hugging Face Spaces

This application is configured for deployment on Hugging Face Spaces with automatic model download:

1. **Build Process**: The `build.py` script automatically downloads the int4 model during Space build
2. **Model Loading**: Uses local model files when available, falls back to Hugging Face download
3. **Caching**: Model files are cached for faster subsequent runs

## 🎛️ Interface Features

### Layout Structure
The interface follows the reference layout with:
- **Title Section**: Main heading and description
- **Information Panels**: Features and model information
- **Input Section**: Context and user input areas
- **Advanced Settings**: Collapsible parameter controls
- **Chat Interface**: Real-time conversation display

### System Prompt
- **Default**: "Tu es TonicIA, un assistant francophone rigoureux et bienveillant."
- **Editable**: Users can customize the system prompt to define the assistant's personality
- **Real-time**: Changes take effect immediately for new conversations

### Generation Parameters
- **Max Length**: Maximum number of tokens to generate (64-2048)
- **Temperature**: Controls randomness in generation (0.01-1.0)
- **Top-p**: Nucleus sampling parameter (0.1-1.0)
- **Enable Thinking**: Enable reasoning mode with thinking tags
- **Advanced Settings**: Collapsible panel for fine-tuning

## 🔧 Technical Details

### Model Loading Strategy
The application uses a smart loading strategy:

1. **Local Check**: First checks if int4 model files exist locally
2. **Local Loading**: If available, loads from `./int4` folder
3. **Fallback Download**: If not available, downloads from Hugging Face
4. **Tokenizer**: Always uses main repo for chat template and configuration

### Build Process
For Hugging Face Spaces deployment:

1. **Build Script**: `build.py` runs during Space build
2. **Model Download**: `download_model.py` downloads int4 model files
3. **Local Storage**: Model files stored in `./int4` directory
4. **Fast Loading**: Subsequent runs use local files

### Chat Template Integration
The application uses the custom chat template from the model, which supports:
- System prompt integration
- User and assistant message formatting
- Thinking mode with `<think>` tags
- Proper conversation flow management

### Memory Optimization
- Uses int4 quantization for reduced memory footprint
- Automatic device detection (CUDA/CPU)
- Efficient tokenization and generation

## 📝 Example Usage

1. **Basic Conversation**:
   - Add context in the system prompt area
   - Type your message in the user input box
   - Click the generate button to start chatting

2. **Customizing System Prompt**:
   - Edit the context in the dedicated text area
   - Changes apply to new messages immediately
   - Example: "Tu es un expert en programmation Python."

3. **Advanced Settings**:
   - Check the "Advanced Settings" checkbox
   - Adjust generation parameters as needed
   - Enable/disable thinking mode

4. **Real-time Chat**:
   - Messages appear in the chat interface
   - Conversation history is maintained
   - Responses are generated using the model's chat template

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure you have sufficient RAM (8GB+ recommended)
   - Check your internet connection for model download
   - Verify all dependencies are installed

2. **Generation Errors**:
   - Try reducing the "Max Length" parameter
   - Adjust temperature and top-p values
   - Check the console for detailed error messages

3. **Performance Issues**:
   - The int4 model is optimized for CPU but may be slower than GPU versions
   - Consider using a machine with more RAM for better performance

4. **System Prompt Issues**:
   - Ensure the system prompt is not too long (max 1000 characters)
   - Check that the prompt follows the expected format

5. **Build Process Issues**:
   - Check that `download_model.py` runs successfully
   - Verify that model files are downloaded to `./int4` directory
   - Ensure sufficient storage space for model files

## 📄 License

This project is licensed under the MIT License. The underlying model is licensed under Apache 2.0.

## 🙏 Acknowledgments

- **Model**: [Tonic/petite-elle-L-aime-3-sft](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft)
- **Base Model**: SmolLM3-3B by HuggingFaceTB
- **Training Data**: legmlai/openhermes-fr
- **Framework**: Gradio, Transformers, PyTorch
- **Layout Reference**: [Tonic/Nvidia-OpenReasoning](https://huggingface.co/spaces/Tonic/Nvidia-OpenReasoning)

## 🔗 Links

- [Model on Hugging Face](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft)
- [Chat Template](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft/blob/main/chat_template.jinja)
- [Original App Reference](https://huggingface.co/spaces/Tonic/Nvidia-OpenReasoning)

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
