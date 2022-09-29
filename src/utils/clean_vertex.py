import argparse
import logging
import sys
import time

import google.cloud.aiplatform as aip
from google.api_core.exceptions import NotFound

VERTEX_REGIONS = {
    'europe-west1', 'europe-west2', 'europe-west3', 'europe-west4',
    'europe-west6', 'europe-west9', 'us-central1', 'us-west1', 'us-west2',
    'us-west4'
}
PROJECT_ID = 'test-rig-349313'
SLEEP_TIME = 1.1


def clean_vertex(
    clean_custom_jobs: bool = False,
    clean_artifacts: bool = False,
    clean_models: bool = False,
    clean_tensorboards: bool = False,
    clean_experiments: bool = False,
    clean_all: bool = False,
    verbosity: bool = False,
):
    if verbosity:
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
    logging.info(f'clean_custom_jobs is set to {clean_custom_jobs}')
    logging.info(f'clean_artifacts is set to {clean_artifacts}')
    logging.info(f'clean_models is set to {clean_models}')
    logging.info(f'clean_tensorboards is set to {clean_tensorboards}')
    logging.info(f'clean_experiments is set to {clean_experiments}')
    logging.info(f'clean_all is set to {clean_all}')
    if clean_all:
        clean_custom_jobs = True
        clean_artifacts = True
        clean_models = True
        clean_tensorboards = True
        clean_experiments = True
    elif not any([
            clean_custom_jobs,
            clean_artifacts,
            clean_models,
            clean_tensorboards,
            clean_experiments,
    ]):
        logging.info(f'No flags set to clean!')
        sys.exit()
    for loc in VERTEX_REGIONS:
        aip.init(
            project=PROJECT_ID,
            location=loc,
        )
        try:
            if clean_custom_jobs:
                for job in aip.CustomJob.list():
                    job.delete()
                    time.sleep(SLEEP_TIME)
                logging.info(f'Custom jobs have been cleaned in {loc}')
            if clean_artifacts:
                for art in aip.Artifact.list():
                    art.delete()
                    time.sleep(SLEEP_TIME)
                logging.info(f'Artifacts have been cleaned in {loc}')
            if clean_models:
                for model in aip.Model.list():
                    model.delete()
                    time.sleep(SLEEP_TIME)
                logging.info(f'Models have been cleaned in {loc}')
            if clean_tensorboards:
                for tb in aip.Tensorboard.list():
                    tb.delete()
                    time.sleep(SLEEP_TIME)
                logging.info(f'Tensorboards have been cleaned in {loc}')
            if clean_experiments:
                for exp in aip.Experiment.list():
                    for exp_run in aip.ExperimentRun.list(experiment=exp):
                        exp_run.delete()
                        time.sleep(SLEEP_TIME)
                    logging.info(
                        f'Experiment runs in {exp.name} have been cleaned in {loc}'
                    )
                    exp.delete()
                    time.sleep(SLEEP_TIME)
                logging.info(f'Experiments have been cleaned in {loc}')
        except NotFound as NotFoundError:
            logging.info(f'{NotFoundError} in {loc}')
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-custom_jobs', '--clean_custom_jobs', action='store_true')
    parser.add_argument('-artifacts', '--clean_artifacts', action='store_true')
    parser.add_argument('-models', '--clean_models', action='store_true')
    parser.add_argument('-tensorboards', '--clean_tensorboards', action='store_true')
    parser.add_argument('-experiments', '--clean_experiments', action='store_true')
    parser.add_argument('-all', '--clean_all', action='store_true')
    parser.add_argument('-v', '--verbosity', action='store_true')
    args = parser.parse_args()
    clean_vertex(**vars(args))
