import os
import pandas as pd
import PyPDF2
from docx import Document
import json
from typing import List, Dict
import email
from email import policy
from email.parser import BytesParser
import logging

class DataProcessor:
    """
    A class to process various document formats for LLM training
    """

    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.supported_formats = ['.txt', '.pdf', '.docx', '.eml']
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logger for the data processor"""
        logger = logging.getLogger('DataProcessor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all documents from the data directory

        Returns:
            List of dictionaries containing document content and metadata
        """
        documents = []

        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()

                if file_extension in self.supported_formats:
                    try:
                        content = self._process_file(file_path, file_extension)
                        if content:
                            documents.append({
                                'content': content,
                                'source': file_path,
                                'type': file_extension
                            })
                            self.logger.info(f"Processed file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error processing file {file_path}: {str(e)}")

        return documents

    def _process_file(self, file_path: str, file_extension: str) -> str:
        """
        Process a single file based on its extension

        Args:
            file_path: Path to the file
            file_extension: File extension

        Returns:
            Extracted text content
        """
        if file_extension == '.txt':
            return self._process_txt(file_path)
        elif file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension == '.docx':
            return self._process_docx(file_path)
        elif file_extension == '.eml':
            return self._process_email(file_path)
        else:
            return ""

    def _process_txt(self, file_path: str) -> str:
        """Process TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Error reading TXT file {file_path}: {str(e)}")
            return ""

    def _process_pdf(self, file_path: str) -> str:
        """Process PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            self.logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            return ""

    def _process_docx(self, file_path: str) -> str:
        """Process DOCX files"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error reading DOCX file {file_path}: {str(e)}")
            return ""

    def _process_email(self, file_path: str) -> str:
        """Process EML email files"""
        try:
            with open(file_path, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)

            # Extract email content
            text = f"Subject: {msg['Subject']}\n"
            text += f"From: {msg['From']}\n"
            text += f"To: {msg['To']}\n"
            text += f"Date: {msg['Date']}\n\n"

            # Get body content
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_content()
                        break
            else:
                text += msg.get_content()

            return text
        except Exception as e:
            self.logger.error(f"Error reading email file {file_path}: {str(e)}")
            return ""

    def chunk_documents(self, documents: List[Dict[str, str]], chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, str]]:
        """
        Split documents into chunks for better processing

        Args:
            documents: List of documents
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            List of chunked documents
        """
        chunks = []

        for doc in documents:
            content = doc['content']
            source = doc['source']

            # Split content into chunks
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i:i + chunk_size]
                if len(chunk_content) > 0:  # Only add non-empty chunks
                    chunks.append({
                        'content': chunk_content,
                        'source': source,
                        'chunk_index': len(chunks)
                    })

        return chunks

# Example usage
if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    # Initialize data processor
    processor = DataProcessor("./data")

    # Load documents
    documents = processor.load_documents()
    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    chunks = processor.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")