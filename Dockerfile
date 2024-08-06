# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Accept the API_KEY argument during build
ARG API_KEY

# Create or overwrite .env file with the API_KEY
RUN echo "OPENAI_API_KEY=$API_KEY" > .env

# Update the package lists and install build-essential
RUN apt-get update && apt-get install -y build-essential

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8000

# Run the command to start uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
