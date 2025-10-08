# Use an official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy current directory contents into the container
COPY . .

# Install dependencies (if you have requirements.txt)
RUN pip install --no-cache-dir -r requirements1.txt

# Expose the port
Expose 8501

#Command to run the app
CMD ["streamlit","run","streamapp.py","--server.port=8501","--server.address=0.0.0.0"]
