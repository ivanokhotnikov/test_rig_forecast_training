import logging
import time

import google.cloud.aiplatform as aip
from google.api_core.exceptions import NotFound

logging.basicConfig(encoding='utf-8', level=logging.INFO)
VERTEX_REGIONS = {
    'europe-west1', 'europe-west2', 'europe-west3', 'europe-west4',
    'europe-west6', 'europe-west9', 'us-central1', 'us-west1', 'us-west2',
    'us-west4'
}
PROJECT_ID = 'test-rig-349313'

if __name__ == '__main__':
    for loc in VERTEX_REGIONS:
        aip.init(
            project=PROJECT_ID,
            location=loc,
        )
        try:
            for job in aip.CustomJob.list():
                job.delete()
                time.sleep(1.1)
            for art in aip.Artifact.list():
                art.delete()
                time.sleep(1.1)
            for model in aip.Model.list():
                model.delete()
                time.sleep(1.1)
            for tb in aip.Tensorboard.list():
                tb.delete()
                time.sleep(1.1)
            for experiment in aip.Experiment.list():
                experiment.delete()
                time.sleep(1.1)
        except NotFound as NotFoundError:
            logging.info(f'{NotFoundError} in {loc}')
            continue
