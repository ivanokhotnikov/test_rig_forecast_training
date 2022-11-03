FROM python:3.10-slim
COPY /configs /app/config
COPY /src /app/src
RUN pip install --no-cache-dir -r app/config/requirements.txt
WORKDIR /app
ENTRYPOINT ["python", "src/training.py"]