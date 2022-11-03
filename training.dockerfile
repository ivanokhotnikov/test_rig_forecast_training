FROM python:3.10-slim
COPY /configs/requirements.txt app/config/requirements.txt
RUN pip install --no-cache-dir -r app/config/requirements.txt
COPY /src /app/src
COPY $HOME/.config/gcloud/application_default_credentials.json /app/key.json
ENV GOOGLE_APPLICATION_CREDENTIALS="app/key.json"
WORKDIR /app
ENTRYPOINT ["python", "src/training.py"]