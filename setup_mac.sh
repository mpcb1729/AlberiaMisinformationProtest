#!/bin/bash
echo "=== Misinformation Protest Simulation Setup (Mac/Linux) ==="

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Python venv creation failed. Make sure Python 3.10+ is installed."
    exit 1
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the simulation:"
echo "  source venv/bin/activate"
echo "  python main.py --no-llm --save-frames --steps 30"
echo ""
echo "To use LLM (requires Ollama):"
echo "  1. Install Ollama: https://ollama.com"
echo "  2. Run: ollama pull qwen3:8b"
echo "  3. Run: python main.py --save-frames"
