import os

from constants import FEATURES_BUCKET, PIPELINES_BUCKET_URI


def download_tb_logs():
    from json import loads

    features_blob = FEATURES_BUCKET.get_blob('forecast_features.json')
    features = loads(features_blob.download_as_text())

    for feature in features:
        os.system(
            f'gsutil -m cp -r {PIPELINES_BUCKET_URI}/tensorboards/{feature} ../logs'
        )


if __name__ == '__main__':
    download_tb_logs()
