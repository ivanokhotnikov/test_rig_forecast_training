import time
import logging

import google.cloud.aiplatform as aip
from google.api_core.exceptions import NotFound

logging.basicConfig(encoding='utf-8', level=logging.INFO)
VERTEX_REGIONS = {
    'us-west1', 'europe-west4', 'us-central1', 'australia-southeast1',
    'europe-west3', 'asia-south1', 'europe-west9', 'us-west2',
    'northamerica-northeast2', 'us-east4', 'asia-east1', 'asia-east2',
    'us-west4', 'northamerica-northeast1', 'asia-northeast1', 'europe-west6',
    'southamerica-east1', 'europe-west1', 'us-east1', 'asia-southeast1',
    'asia-northeast3', 'europe-west2'
}

if __name__ == '__main__':
    for loc in VERTEX_REGIONS:
        aip.init(
            project='test-rig-349313',
            location=loc,
        )
        try:
            for job in aip.CustomJob.list():
                job.delete()
                time.sleep(1.5)
            for art in aip.Artifact.list():
                art.delete()
                time.sleep(1.5)
            for model in aip.Model.list():
                model.delete()
                time.sleep(1.5)
            for tb in aip.Tensorboard.list():
                tb.delete()
                time.sleep(1.5)
            for experiment in aip.Experiment.list():
                experiment.delete()
                time.sleep(1.5)
        except NotFound as NotFoundError:
            logging.info(f'{NotFoundError} in {loc}')
            continue