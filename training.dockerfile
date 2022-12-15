FROM python:3.10-slim
COPY /configs/prod.txt /app/config/prod.txt
COPY /src /app/src
RUN pip install --no-cache-dir -r app/config/prod.txt
WORKDIR /app
ENTRYPOINT ["make", "default_run"]