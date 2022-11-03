FROM python:3.10-slim
COPY /configs/requirements.txt app/config/requirements.txt
RUN pip install --no-cache-dir -r app/config/requirements.txt
COPY /src /app/src
RUN gsutil cp gs://gcloud_keys/key.json /app/gcloud_keys/key.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gcloud_keys/key.json"
WORKDIR /app
ENTRYPOINT ["python", "src/training.py"]