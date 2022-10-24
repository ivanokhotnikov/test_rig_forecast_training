from kfp.v2.dsl import Artifact, Dataset, Output, component


@component(
    base_image='python:3.10-slim',
    packages_to_install=['pandas', 'openpyxl'],
)
def read_raw_data(
    interim_data: Output[Dataset],
    raw_features: Output[Artifact],
) -> None:
    """Read raw data files from the GCS bucket, specified by `bucket_name`. Uploads the combined data frame to the interim data directory in the GCS bucket.

    Args:
        interim_data (Output[Dataset]): Interim data
        all_features (Output[Artifact]): Raw features artifact
    """
    import gc
    import json
    import logging
    import os
    import re

    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    final_df = pd.DataFrame()
    raw_data_path = os.path.join('gcs', 'test_rig_raw_data')
    units = []
    for file in os.listdir(raw_data_path):
        logging.info(f'Reading {file} from {raw_data_path}')
        try:
            if file.endswith('.csv') and 'RAW' in file:
                current_df = pd.read_csv(
                    os.path.join(raw_data_path, file),
                    header=0,
                    index_col=False,
                )
            elif (file.endswith('.xlsx')
                  or file.endswith('.xls')) and 'RAW' in file:
                current_df = pd.read_excel(
                    os.path.join(raw_data_path, file),
                    header=0,
                    index_col=False,
                )
            else:
                logging.info(f'{file} is not a valid raw data file')
                continue
        except:
            logging.info(f'Cannot read {file}')
            continue
        logging.info(f'{file} has been read')
        try:
            unit = int(re.split(r'_|-|/', file)[0][4:].lstrip('HYD0'))
        except ValueError as err:
            logging.info(f'{err}\n. Cannot parse unit from {file}')
            continue
        units.append(unit)
        current_df['UNIT'] = unit
        current_df['TEST'] = int(units.count(unit))
        final_df = pd.concat((final_df, current_df), ignore_index=True)
        del current_df
        gc.collect()
    try:
        final_df.sort_values(by=['UNIT', 'TEST'],
                             inplace=True,
                             ignore_index=True)
        logging.info(f'Final dataframe sorted')
    except:
        logging.info('Cannot sort dataframe')
    final_df.to_csv(
        interim_data.path + '.csv',
        index=False,
    )
    logging.info('Interim dataframe uploaded to the piepline metadata store')
    final_df.to_csv(
        os.path.join('gcs', 'test_rig_interim_data', 'interim_data.csv'),
        index=False,
    )
    logging.info('Interim dataframe uploaded to the interim data storage')
    with open(raw_features.path + '.json', 'w') as features_file:
        json.dump(final_df.columns.to_list(), features_file)
    logging.info('Raw features uploaded to the pipeline metadata store')
    with open(
            os.path.join('gcs', 'test_rig_features',
                         'raw_features.json'), 'w') as features_file:
        json.dump(final_df.columns.to_list(), features_file)
    logging.info('Raw features uploaded to the featues store')
