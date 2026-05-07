@echo off
echo === Misinformation Protest Simulation Setup (Windows) ===

python -m venv venv
if errorlevel 1 (
    echo ERROR: Python venv creation failed. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === Setup complete! ===
echo.
echo To run the simulation:
echo   venv\Scripts\activate
echo   python main.py --no-llm --save-frames --steps 30
echo.
echo To use LLM (requires Ollama):
echo   1. Install Ollama: https://ollama.com
echo   2. Run: ollama pull qwen3:8b
echo   3. Run: python main.py --save-frames
echo.
pause
