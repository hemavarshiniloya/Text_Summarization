# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the container
COPY . .

# Expose the port that the app runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "your_app_file.py", "--server.port=8501", "--server.address=0.0.0.0"]
