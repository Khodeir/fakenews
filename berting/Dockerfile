# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /usr/src/
RUN mkdir /usr/src/models
RUN mkdir /usr/src/condensed_articles

# Copy the current directory contents into the container at /app
COPY ./do_inference.py /usr/src/
COPY ./data_providers.py /usr/src/
COPY ./requirements.txt /usr/src/
COPY ./bert_base_uncased_sts-b /usr/src/models/bert_base_uncased_sts-b
COPY ./condensed_v2_base_dup_2e-5 /usr/src/models/condensed_v2_base_dup_2e-5

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md

# Define environment variable
ENV NAME fakenews

# Run app.py when the container launches
CMD [ "python3", "/usr/src/do_inference.py" ]

