import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from src.data_processor import DataProcessor

class QuestionAnsweringSystem:
    """
    A question-answering system using a fine-tuned LLM and retrieval-augmented generation
    """

    def __init__(self, model_path: str = "./models", config_path: str = "./config.json"):
        """
        Initialize the QA system

        Args:
            model_path: Path to the trained model
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize components
        self.model_path = model_path
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.index = None
        self.documents = []

        # Load components
        self._load_embedding_model()
        self._load_llm()

    def _load_embedding_model(self):
        """Load the embedding model for document retrieval"""
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])

    def _load_llm(self):
        """Load the fine-tuned LLM"""
        print("Loading LLM...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto"
        )

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index documents for fast retrieval

        Args:
            documents: List of documents to index
        """
        self.documents = documents

        # Generate embeddings for all documents
        print("Generating document embeddings...")
        document_texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_model.encode(document_texts, show_progress_bar=True)

        # Create FAISS index
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        print(f"Indexed {len(documents)} documents")

    def _retrieve_relevant_documents(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of (document_index, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar documents
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        # Return results as list of (index, score) tuples
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    def _format_prompt(self, query: str, context: str) -> str:
        """
        Format the prompt for the LLM

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""<s>[INST] Answer the question based on the context provided.
Context: {context}

Question: {query}

Answer: [/INST]"""
        return prompt

    def answer_question(self, query: str, k: int = 3) -> str:
        """
        Answer a question using the trained model and retrieved context

        Args:
            query: Question to answer
            k: Number of context documents to retrieve

        Returns:
            Answer to the question
        """
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_documents(query, k)

        # Extract context from documents
        context_parts = []
        for doc_idx, score in relevant_docs:
            if doc_idx < len(self.documents):
                context_parts.append(self.documents[doc_idx]['content'])

        context = "\n\n".join(context_parts)

        # Format prompt
        prompt = self._format_prompt(query, context)

        # Generate answer using LLM
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

        # Decode the generated answer
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part (remove the prompt)
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()

        return answer

    def batch_answer_questions(self, questions: List[str], k: int = 3) -> List[str]:
        """
        Answer multiple questions

        Args:
            questions: List of questions
            k: Number of context documents to retrieve per question

        Returns:
            List of answers
        """
        return [self.answer_question(q, k) for q in questions]

# Example usage
if __name__ == "__main__":
    # Initialize QA system
    qa_system = QuestionAnsweringSystem()

    # Load and index documents (this would typically be done once during setup)
    processor = DataProcessor("./data")
    documents = processor.load_documents()
    chunks = processor.chunk_documents(documents)

    # Index documents for retrieval
    qa_system.index_documents(chunks)

    # Example question
    question = "What is the main topic of the documents?"
    answer = qa_system.answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}")