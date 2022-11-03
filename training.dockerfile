FROM python:3.10-slim
COPY /configs/requirements.txt app/config/requirements.txt
RUN pip install --no-cache-dir -r app/config/requirements.txt
COPY /src /app/src
WORKDIR /app
ENTRYPOINT ["python", "src/training.py"]