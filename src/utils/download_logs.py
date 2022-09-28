def download_logs():
    import os
    from json import loads

    from google.cloud.storage import Client

    DATA_BUCKET_NAME = 'test_rig_data'
    PIPELINES_BUCKET_NAME = 'test_rig_pipelines'
    PIPELINES_BUCKET_URI = f'gs://{PIPELINES_BUCKET_NAME}'
    storage_client = Client()
    data_bucket = storage_client.get_bucket(DATA_BUCKET_NAME)

    features_blob = data_bucket.get_blob('final_features.json')
    features = loads(features_blob.download_as_text())

    for feature in features:
        os.system(f'gsutil -m cp -r {PIPELINES_BUCKET_URI}/tb/{feature} logs')


if __name__ == '__main__':
    download_logs()