# Use the official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of your project files into the container
COPY . .

# This line to tell Python where to find the 'src' folder
ENV PYTHONPATH=/app

# Expose the Hugging Face port
EXPOSE 7860

# Command to run your Flask application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]