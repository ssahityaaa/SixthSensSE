# Use the official Python image as a base image
FROM python:3.9

# Set environment variables
ENV PORT 8080

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
COPY . /templates
COPY . /static
COPY . /handwriting_recognition_model2.h5

# Install dependencies
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python", "app.py"]
