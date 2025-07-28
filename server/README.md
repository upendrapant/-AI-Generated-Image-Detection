# Server

This directory contains the API server for model inference.

## Quick Start

1. Install Python dependencies (from the main project root):
   ```sh
   pip install -r ../requirements.txt
   ```
2. (Optional) Install Node.js dependencies if needed:
   ```sh
   npm install
   ```
3. Run the API server:
   - For FastAPI:
     ```sh
     uvicorn api:app --reload
     ```
   - For Flask:
     ```sh
     python api.py
     ```

See the main [README](../README.md) for more details. 