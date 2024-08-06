## Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use ⁠ venv\Scripts\activate ⁠

# Install the dependencies
pip install -r requirements.txt

# Run the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000
