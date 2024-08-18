# Create a virtual environment
`python -m venv venv`

# Activate venv depending on OS …
## On Mac
`source venv/bin/activate`

## On Windows
`venv\Scripts\activate`

# Install the dependencies
`pip install -r requirements.txt`

# Run the FastAPI application
`uvicorn main:app --host 0.0.0.0 --port 8000`

# Ruf die Swagger Doku auf
http://127.0.0.1:8000/docs