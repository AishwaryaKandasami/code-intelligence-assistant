#!/bin/bash
# Code Intelligence Assistant - Installation Script

echo "=================================="
echo "Code Intelligence Assistant Setup"
echo "=================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠ Ollama not found"
    echo "Please install from: https://ollama.ai"
    read -p "Press enter after installing Ollama..."
fi

echo "✓ Ollama found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-prod.txt

echo "✓ Dependencies installed"

# Pull Ollama model
echo ""
echo "Pulling Ollama model (this may take a few minutes)..."
ollama pull qwen2.5-coder:7b

echo "✓ Model downloaded"

# Check .env file
if [ ! -f .env ]; then
    echo ""
    echo "⚠ .env file not found"
    echo "Creating template..."
    cat > .env << 'EOF'
# OpenRouter Configuration (optional)
OPENROUTER_API_KEY=your_key_here
LLM_MODEL=meta-llama/llama-3.1-8b-instruct

# Ollama Configuration
OLLAMA_MODEL=qwen2.5-coder:7b

# Feature Flags
HYBRID_MODE=true
EOF
    echo "✓ .env template created - please add your OpenRouter API key"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import ollama; import sentence_transformers; print('✓ All imports successful')"

echo ""
echo "=================================="
echo "✅ Installation Complete!"
echo "=================================="
echo ""
echo "To use the system:"
echo "  1. Activate: source venv/bin/activate  (or venv\\Scripts\\activate on Windows)"
echo "  2. Review code: python review.py path/to/code.py"
echo "  3. Batch review: python batch_review.py path/to/directory"
echo ""
echo "For help: python review.py --help"