FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
ARG MODEL_PATH
ARG VOCAB_PATH=export/vocab.txt

COPY reading/ /ava/reading/
COPY utils/ /ava/utils/
COPY ${MODEL_PATH} /ava/reading/model.pt
COPY ${VOCAB_PATH} /ava/data/vocab.txt
RUN pip install -r /ava/reading/requirements.txt
WORKDIR /ava

ENV PYTHONPATH=/ava/

ENV JSON_PATH=/usr/local/dataset/metadata.json
ENV DATA_DIR=/usr/local/dataset/articles
ENV RESUME_FROM=/ava/reading/model.pt
ENV DF_PATH=/usr/local/predictions.txt
CMD python reading/predict.py


