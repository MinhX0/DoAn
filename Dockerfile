# Use an official Python image
FROM python:3.12.10

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y gcc

# Copy requirements (create this file if you don't have it)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "stock_prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]