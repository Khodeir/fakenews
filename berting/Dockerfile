# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./condense_articles_docker.py /app
COPY ./data_providers.py /app
COPY ./run_lang_model.py /app
COPY ./start.sh /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Define environment variable
ENV NAME fakenews

RUN chmod u+x start.sh

# Run app.py when the container launches
CMD ./start.sh
