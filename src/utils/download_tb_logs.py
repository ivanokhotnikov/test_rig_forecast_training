import os

from utils.constants import DATA_BUCKET_NAME, STORAGE_CLIENT, PIPELINES_BUCKET_URI


def download_tb_logs():
    from json import loads

    data_bucket = STORAGE_CLIENT.get_bucket(DATA_BUCKET_NAME)

    features_blob = data_bucket.get_blob('final_features.json')
    features = loads(features_blob.download_as_text())

    for feature in features:
        os.system(f'gsutil -m cp -r {PIPELINES_BUCKET_URI}/tb/{feature} logs')


if __name__ == '__main__':
    download_tb_logs()
