FROM python:3.10-slim
COPY . /app
RUN pip install --no-cache-dir -r app/configs/prod.txt
WORKDIR /app
ENTRYPOINT ["make", "default_run"]