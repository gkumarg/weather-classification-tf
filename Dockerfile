# Use an official Python runtime as a parent image
FROM python:3.9.13-slim

RUN pip install pipenv

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the dependencies file to the working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

# Copy the current directory contents into the container at /usr/src/app
COPY ["predict.py", "MobileNetv2.tflite", "./"]

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME Weather

# Run predict.py when the container launches
# CMD ["python", "predict.py"]

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "predict:app"]
