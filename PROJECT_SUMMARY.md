# LLM Document Processing and QA System - Project Summary

## Overview

This project provides a complete solution for training a custom Language Model (LLM) on your personal documents and emails, enabling you to ask questions and receive accurate answers based on your data. The system combines document processing, model fine-tuning capabilities, and a retrieval-augmented question-answering system.

## Key Components

### 1. Data Processor (`src/data_processor.py`)
Processes various document formats:
- PDF documents
- Microsoft Word documents (DOCX)
- Plain text files (TXT)
- Email files (EML)

Features:
- Extracts text content from multiple document formats
- Handles document chunking for better processing
- Maintains document metadata and source tracking

### 2. Model Trainer (`src/model_trainer.py`)
Handles fine-tuning of the LLM using Parameter-Efficient Fine-Tuning (LoRA):
- Uses Mistral 7B as the base model
- Implements 8-bit quantization for memory efficiency
- Configurable training parameters

### 3. Question-Answering System (`src/qa_system.py`)
Retrieval-Augmented Generation (RAG) system:
- Embeds documents using sentence transformers
- Uses FAISS for efficient similarity search
- Generates answers using the fine-tuned LLM

### 4. Main Application (`src/main.py`)
Command-line interface for the entire system:
- Document processing mode
- Model training mode
- Question-answering mode
- Interactive QA session mode

## How It Works

### Phase 1: Document Processing
1. User places documents in the `data/` directory
2. System processes documents using `DataProcessor`
3. Documents are chunked into manageable pieces
4. Processed data is saved to `processed/` directory

### Phase 2: Model Training (Optional)
1. System loads pre-trained Mistral 7B model
2. Applies LoRA fine-tuning for efficient training
3. Trains model on user's document chunks
4. Saves trained model to `models/` directory

### Phase 3: Question Answering
1. System indexes documents using embeddings
2. User asks questions through CLI or interactive mode
3. Relevant documents are retrieved using FAISS
4. LLM generates answers based on retrieved context

## System Requirements

### Hardware
- **Minimum**: 16GB RAM, modern CPU
- **Recommended**: 32GB+ RAM, GPU with 12GB+ VRAM

### Software
- Python 3.8+
- Required Python packages (see `requirements.txt`)

## Usage Workflow

1. **Setup**:
   ```
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place documents in `data/` directory

3. **Process Documents**:
   ```
   python src/main.py --mode process
   ```

4. **Train Model (Optional)**:
   ```
   python src/main.py --mode train
   ```

5. **Ask Questions**:
   ```
   python src/main.py --mode qa --question "Your question"
   ```

6. **Interactive Session**:
   ```
   python src/main.py --mode interactive
   ```

## Privacy and Security

- All processing happens locally on your machine
- No data is sent to external servers
- Documents remain private and secure

## Advantages

1. **Custom Knowledge Base**: Train on your specific documents and emails
2. **Privacy-Focused**: Everything runs locally
3. **Flexible Formats**: Supports multiple document types
4. **Efficient Training**: Uses LoRA for memory-efficient fine-tuning
5. **Accurate Retrieval**: RAG system ensures relevant context
6. **Easy to Use**: Simple command-line interface

## Limitations

1. **Resource Intensive**: Requires significant RAM and potentially GPU
2. **Training Time**: Fine-tuning can take hours depending on data size
3. **Model Size**: Mistral 7B requires substantial storage space

## Future Enhancements

1. Support for additional document formats
2. Web interface for easier interaction
3. Integration with email clients for automatic email processing
4. Advanced chunking strategies for better context retention
5. Multi-model support for different use cases

## Conclusion

This system provides a comprehensive solution for creating a personalized question-answering system based on your own documents and emails. With its modular design and clear separation of concerns, it can be easily extended and customized for specific needs.