# Custom LLM Document Processing and QA System

This project provides a complete solution for training a custom LLM on your own documents and emails, and then using it to answer questions about that data.

## Features

- Process multiple document formats (PDF, DOCX, TXT, EML)
- Fine-tune a Mistral 7B model on your data using LoRA
- Question-answering system with retrieval-augmented generation
- Interactive QA interface

## Prerequisites

- Python 3.8 or higher
- 16GB+ RAM (32GB recommended)
- GPU with 12GB+ VRAM recommended (optional but faster)

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Your Data

Place your documents (PDF, DOCX, TXT, EML files) in the `data/` directory.

### 2. Process Documents

```bash
python src/main.py --mode process
```

This will process all documents and create chunks for training.

### 3. Train Model (Optional)

To fine-tune the model on your data:

```bash
python src/main.py --mode train
```

Note: This step requires significant computational resources and time.

### 4. Question Answering

To answer questions about your documents:

```bash
python src/main.py --mode qa --question "Your question here"
```

### 5. Interactive QA Session

For an interactive session:

```bash
python src/main.py --mode interactive
```

### 6. Using the Jupyter Notebook

You can also experiment with the system using the Jupyter notebook:

```bash
jupyter notebook notebooks/experiment.ipynb
```

## Configuration

The `config.json` file contains various settings for the model and training process. You can modify these settings according to your needs.

## Project Structure

```
llm_project/
├── config.json          # Configuration file
├── requirements.txt     # Python dependencies
├── data/                # Your documents go here
├── processed/           # Processed data (created automatically)
├── models/              # Trained models (created after training)
├── src/                 # Source code
│   ├── data_processor.py # Document processing
│   ├── model_trainer.py  # Model training
│   ├── qa_system.py      # Question answering system
│   └── main.py           # Main application
└── notebooks/           # Jupyter notebooks
    └── experiment.ipynb   # Experiment notebook
```

## How It Works

1. **Document Processing**: The system reads and processes various document formats, extracting text content.

2. **Chunking**: Large documents are split into smaller chunks for better processing.

3. **Embedding**: Documents are embedded using a sentence transformer for efficient retrieval.

4. **Fine-tuning (Optional)**: The Mistral 7B model can be fine-tuned on your data using LoRA for better performance.

5. **Retrieval-Augmented Generation**: When answering questions, relevant documents are retrieved and used as context for the LLM.

6. **Question Answering**: The LLM generates answers based on the retrieved context.

## Notes

- The system uses the Mistral 7B model by default, which provides a good balance between performance and resource requirements.
- For privacy, all processing happens locally on your machine.
- Fine-tuning requires significant computational resources and time. If you don't have a powerful GPU, consider using the pre-trained model with retrieval-augmented generation.