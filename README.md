# HarryPotterGPT: Building a Language Model from Scratch

![HarryPotterGPT](https://img.shields.io/badge/HarryPotterGPT-v1.0-purple)
![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🧙‍♂️ About The Project

HarryPotterGPT is an educational implementation of a GPT-style language model built from scratch using PyTorch. The project trains a transformer-based model on Harry Potter books to generate text in the style of J.K. Rowling's famous series. This repository serves as both a practical implementation and an in-depth tutorial on building modern NLP architectures.

The entire process - from raw text processing to deploying a user interface - is implemented step-by-step in a modular, well-documented codebase.

## 👨‍💻 Author

**Camilo Vega** - AI Consultant and Professor
- [LinkedIn Profile](https://www.linkedin.com/in/camilo-vega-169084b1/)

## 📋 Features

- **🔄 Complete ML Pipeline**: Full implementation from data processing to model serving
- **📚 Multi-format Support**: Process text from PDF, TXT, and DOCX files
- **🔤 Custom Tokenization**: Uses SentencePiece for subword tokenization with vocabulary adaptation
- **🧠 Transformer Architecture**: Implementation of the GPT (decoder-only transformer) architecture
- **📈 Training Optimizations**: Includes learning rate scheduling, early stopping, and validation
- **🖥️ Interactive UI**: Gradio web interface for text completion
- **📝 Extensive Documentation**: Detailed comments and visualization of model architecture
- **🛡️ Error Handling**: Robust recovery mechanisms for various failure scenarios

## 🚀 Running the Script

This repository includes a fully functional Python script (`HarryPotterGPT.ipynb`/`.py`) that implements the entire pipeline from data processing to model generation. The script is designed to be both educational and practical, with extensive comments explaining each step of the process.

### How to Use the Script

1. **Open in Google Colab or Jupyter**: The notebook format makes it easy to run each phase separately and observe the results.
   ```
   # If using Colab, you can open directly from GitHub
   # File > Open notebook > GitHub > Paste repository URL
   ```

2. **Run All Phases**: Execute the entire pipeline from start to finish:
   ```python
   # Simply run all cells in order to:
   # - Process your Harry Potter books
   # - Train a custom tokenizer
   # - Build and train a GPT model
   # - Create an interactive text generation interface
   ```

3. **Learning Approach**: The script is designed for learning by doing:
   - Each phase includes detailed explanations of what's happening
   - Key components (attention mechanism, tokenization, etc.) are thoroughly documented
   - Implementation details match the theoretical concepts outlined in this README

The script serves as both a practical tool and a comprehensive tutorial, allowing you to understand what's happening at each step of building a language model from scratch.

## 🔍 Transformer Architecture Diagram

```
                         GPT MODEL ARCHITECTURE (DECODER-ONLY)
+----------------------------------------------------------------------------------+
|                                                                                  |
|  ┌─────────────────┐                                                             |
|  │   Input Text    │                                                             |
|  └─────────────────┘                                                             |
|           ↓                                                                      |
|  ┌─────────────────┐     ┌─────────────────┐                                     |
|  │ Token Embedding │ +   │ Position Embed. │ → Dropout                           |
|  └─────────────────┘     └─────────────────┘                                     |
|           ↓                                                                      |
|  ┌───────────────────────────────────────────────────────────┐                   |
|  │                   Transformer Block × N                   │                   |
|  │  ┌────────────────────────────────────────────────────┐   │                   |
|  │  │ ┌─────────────┐                                    │   │                   |
|  │  │ │ Layer Norm  │                                    │   │                   |
|  │  │ └─────────────┘                                    │   │                   |
|  │  │        ↓                                           │   │                   |
|  │  │ ┌─────────────────────────────────────────┐        │   │                   |
|  │  │ │ Multi-Head Self-Attention (with mask)   │        │   │                   |
|  │  │ └─────────────────────────────────────────┘        │   │                   |
|  │  │        ↓                                           │   │                   |
|  │  │ Residual Connection → Dropout                      │   │                   |
|  │  │        ↓                                           │   │                   |
|  │  │ ┌─────────────┐                                    │   │                   |
|  │  │ │ Layer Norm  │                                    │   │                   |
|  │  │ └─────────────┘                                    │   │                   |
|  │  │        ↓                                           │   │                   |
|  │  │ ┌─────────────────────────────────────────┐        │   │                   |
|  │  │ │            Feed Forward Network         │        │   │                   |
|  │  │ │ Linear → GELU → Dropout → Linear → Drop │        │   │                   |
|  │  │ └─────────────────────────────────────────┘        │   │                   |
|  │  │        ↓                                           │   │                   |
|  │  │ Residual Connection → Dropout                      │   │                   |
|  │  └────────────────────────────────────────────────────┘   │                   |
|  └───────────────────────────────────────────────────────────┘                   |
|           ↓                                                                      |
|  ┌─────────────────┐                                                             |
|  │   Layer Norm    │                                                             |
|  └─────────────────┘                                                             |
|           ↓                                                                      |
|  ┌─────────────────┐                                                             |
|  │ Output Projection│ → Logits                                                    |
|  └─────────────────┘                                                             |
+----------------------------------------------------------------------------------+
```

## 🛠️ Technical Details

### 🧩 Pipeline Phases

1. **📦 Dependency Installation**: Setup of required libraries
2. **📚 Library Imports**: Organization of necessary Python modules
3. **📝 Data Processing**: Multi-format text extraction and cleaning
4. **🔤 Tokenizer Training**: Custom SentencePiece tokenizer implementation
5. **⚙️ Model Configuration**: Transformer architecture setup
6. **🧠 Model Architecture**: GPT model implementation (attention mechanism, feed-forward network, etc.)
7. **🏋️ Model Training**: Training with optimization strategies
8. **🖥️ Gradio Interface**: Interactive web UI for model inference

### 🏗️ Architecture Highlights

- **🔤 Tokenization**: SentencePiece Unigram model with special token handling
- **📊 Embedding**: Token and positional embeddings
- **👁️ Transformer Blocks**: Multi-headed self-attention with causal masking and feed-forward networks
- **📊 Pre-Layer Normalization**: Modern transformer architecture with pre-LN
- **📈 Training**: AdamW optimizer with learning rate warmup and cosine decay
- **🎲 Generation**: Top-k sampling with temperature control

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Sufficient RAM for training (8GB+ recommended)
- GPU recommended for faster training (though CPU mode is supported)

### Installation

```bash
# Clone the repository
git clone https://github.com/CamiloVga/HarryPotterGPT.git
cd HarryPotterGPT

# Install dependencies
pip install torch numpy matplotlib pandas tqdm PyPDF2 nltk python-docx sentencepiece gradio
```

### Usage

1. **Prepare your data**: Place Harry Potter books (PDF, TXT, or DOCX format) in the `books` directory.

2. **Run the training pipeline**:
```python
# Run all phases
python harrypottergpt.py
```

3. **Use the model**:
```python
# For just the inference interface
python harrypottergpt.py --mode inference --model_path gpt_harry_potter_trained_final.pt
```

4. **Generate text using the web UI**:
   - Enter a prompt such as "Harry looked at Hermione and"
   - Adjust generation parameters (temperature, max tokens, etc.)
   - Click "Generate Completion"

## 📊 Model Performance

The model performance will depend on:
- Size and quality of your Harry Potter corpus
- Selected hyperparameters (especially model size and training duration)
- Available computational resources

With the default configuration and a complete Harry Potter book series, the model can generate coherent text that captures the style and themes of the original books.

## 📝 Code Example: Text Generation

```python
# Quick example of generating text with a trained model
model.eval()
tokenizer = SentencePieceTokenizer()
tokenizer.load("spm_harry_potter")

prompt = "Harry looked around the"
input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

with torch.no_grad():
    generated_ids = model.generate(
        input_tensor,
        max_new_tokens=100,
        temperature=0.7,
        top_k=40,
        tokenizer=tokenizer
    )

generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
print(generated_text)
```

## 🧠 Educational Value

This repository serves as a comprehensive learning resource for:
- 🔍 NLP preprocessing and tokenization
- 🏗️ Transformer architecture implementation
- 📚 Language model training
- 🔄 Text generation strategies
- 🖥️ Deploying models with a user interface

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgements

- The PyTorch team for their powerful deep learning framework
- The SentencePiece team for their tokenization library
- The Gradio team for the easy-to-use UI framework
- J.K. Rowling for the Harry Potter series that made this project more magical

## ⚠️ Legal Note

This project is for educational purposes only. Users are responsible for ensuring they have the appropriate rights to any text they use for training. The Harry Potter books are copyrighted material, and users should obtain legitimate copies for personal use.

---

Happy coding, and may your text generation be as magical as Hogwarts! ✨
