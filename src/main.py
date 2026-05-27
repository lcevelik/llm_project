import os
import argparse
import json
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.qa_system import QuestionAnsweringSystem

def main():
    parser = argparse.ArgumentParser(description="LLM Document Processing and QA System")
    parser.add_argument("--mode", choices=["process", "train", "qa", "interactive"],
                        help="Mode of operation")
    parser.add_argument("--data-dir", default="./data", help="Directory containing documents")
    parser.add_argument("--model-dir", default="./models", help="Directory for model storage")
    parser.add_argument("--config", default="./config.json", help="Configuration file path")
    parser.add_argument("--question", help="Question for QA mode")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.mode == "process":
        process_documents(args.data_dir)
    elif args.mode == "train":
        train_model(args.data_dir, args.model_dir, args.config)
    elif args.mode == "qa":
        if not args.question:
            print("Please provide a question with --question")
            return
        answer_question(args.question, args.model_dir, args.config)
    elif args.mode == "interactive":
        interactive_qa(args.model_dir, args.config)
    else:
        print("Please specify a mode: process, train, qa, or interactive")

def process_documents(data_dir):
    """Process documents in the data directory"""
    print("Processing documents...")
    processor = DataProcessor(data_dir)
    documents = processor.load_documents()
    chunks = processor.chunk_documents(documents)

    # Save processed data
    os.makedirs("./processed", exist_ok=True)
    with open("./processed/documents.json", "w") as f:
        json.dump(documents, f, indent=2)

    with open("./processed/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
    print("Documents saved to ./processed/")

def train_model(data_dir, model_dir, config_path):
    """Train the model on processed documents"""
    print("Training model...")

    # Load processed data
    try:
        with open("./processed/chunks.json", "r") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Processed data not found. Run 'process' mode first.")
        return

    # Initialize trainer
    trainer = ModelTrainer(config_path)
    trainer.load_model_and_tokenizer()

    # Prepare dataset
    dataset = trainer.prepare_dataset(chunks)

    # Train model
    trainer.train(dataset)

    print("Model training completed!")

def answer_question(question, model_dir, config_path):
    """Answer a single question"""
    print(f"Answering question: {question}")

    # Load processed data
    try:
        with open("./processed/chunks.json", "r") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Processed data not found. Run 'process' mode first.")
        return

    # Initialize QA system
    qa_system = QuestionAnsweringSystem(model_dir, config_path)
    qa_system.index_documents(chunks)

    # Get answer
    answer = qa_system.answer_question(question)
    print(f"Answer: {answer}")

def interactive_qa(model_dir, config_path):
    """Interactive QA session"""
    print("Starting interactive QA session...")
    print("Type 'quit' to exit")

    # Load processed data
    try:
        with open("./processed/chunks.json", "r") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Processed data not found. Run 'process' mode first.")
        return

    # Initialize QA system
    qa_system = QuestionAnsweringSystem(model_dir, config_path)
    qa_system.index_documents(chunks)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'quit':
            break

        if question:
            answer = qa_system.answer_question(question)
            print(f"Answer: {answer}")

if __name__ == "__main__":
    main()