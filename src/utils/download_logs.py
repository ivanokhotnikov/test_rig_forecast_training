import os
from google.cloud.storage import Client
from json import loads

DATA_BUCKET_NAME = 'test_rig_data'
DATA_BUCKET_URI = f'gs://{DATA_BUCKET_NAME}'
PIPELINES_BUCKET_NAME = 'test_rig_pipelines'
PIPELINES_BUCKET_URI = f'gs://{PIPELINES_BUCKET_NAME}'

storage_client = Client()
data_bucket = storage_client.get_bucket(DATA_BUCKET_NAME)
pipelines_bucket = storage_client.get_bucket(PIPELINES_BUCKET_NAME)

features_blob = data_bucket.get_blob('final_features.json')
features = loads(features_blob.download_as_text())

for feature in features:
    os.system(f'gsutil cp -r {PIPELINES_BUCKET_URI}/{feature} logs')