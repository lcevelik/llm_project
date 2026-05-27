"""
Test script to verify the installation of the LLM system
"""

import os
import sys

def test_installation():
    """Test if all required packages are installed"""
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'trl',
        'bitsandbytes',
        'accelerate',
        'sentence_transformers',
        'langchain',
        'faiss_cpu',
        'PyPDF2',
        'python_docx'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'faiss_cpu':
                import faiss
            elif package == 'PyPDF2':
                import PyPDF2
            elif package == 'python_docx':
                from docx import Document
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run 'pip install -r requirements.txt' to install missing packages")
        return False
    else:
        print("\n✓ All required packages are installed")
        return True

def test_directories():
    """Test if required directories exist"""
    required_dirs = ['data', 'processed', 'models', 'src']

    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory '{directory}' exists")
        else:
            print(f"✗ Directory '{directory}' is missing")
            if directory in ['data', 'processed', 'models']:
                print(f"  Creating '{directory}' directory...")
                os.makedirs(directory, exist_ok=True)
                print(f"  ✓ Created '{directory}' directory")
    return True

def test_src_files():
    """Test if required source files exist"""
    required_files = [
        'src/data_processor.py',
        'src/model_trainer.py',
        'src/qa_system.py',
        'src/main.py'
    ]

    for file in required_files:
        if os.path.exists(file):
            print(f"✓ File '{file}' exists")
        else:
            print(f"✗ File '{file}' is missing")
            return False

    return True

def main():
    """Main test function"""
    print("Testing LLM Document Processing and QA System Installation")
    print("=" * 60)

    # Test packages
    print("\n1. Testing Python packages:")
    packages_ok = test_installation()

    # Test directories
    print("\n2. Testing directories:")
    dirs_ok = test_directories()

    # Test source files
    print("\n3. Testing source files:")
    src_ok = test_src_files()

    # Summary
    print("\n" + "=" * 60)
    if packages_ok and dirs_ok and src_ok:
        print("✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Place your documents (PDF, DOCX, TXT, EML) in the 'data' directory")
        print("2. Run 'python src/main.py --mode process' to process your documents")
        print("3. Run 'python src/main.py --mode interactive' for interactive QA")
    else:
        print("✗ Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()