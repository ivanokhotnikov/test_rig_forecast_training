import os
from kfp.v2.dsl import Dataset, Output, Artifact, component


@component(
    base_image='python:3.10-slim',
    packages_to_install=['pandas', 'openpyxl'],
    output_component_file=os.path.join('configs', 'read_raw_gcs.yaml'),
)
def read_raw_data(
    data_bucket_name: str,
    interim_data: Output[Dataset],
    all_features: Output[Artifact],
) -> None:
    """Read raw data files from the GCS bucket, specified by `bucket_name`. Uploads the combined data frame to the interim data directory in the GCS bucket.

    Args:
        data_bucket_name (str): GCS data bucket
        interim_data (Output[Dataset]): Interim data
        allfeatures (Output[Artifact]): Raw features artifact
    """
    import os
    import gc
    import json
    import logging

    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    final_df = pd.DataFrame()
    raw_data_path = os.path.join('gcs', data_bucket_name, 'raw')
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
            logging.info(f'{file} was read!')
            final_df = pd.concat((final_df, current_df), ignore_index=True)
            del current_df
            gc.collect()
        except:
            logging.info(f'Can\'t read {file}!')
            continue
    final_df.to_csv(
        interim_data.path + '.csv',
        index=False,
    )
    interim_data_path = os.path.join('gcs', data_bucket_name, 'interim')
    final_df.to_csv(
        os.path.join(interim_data_path, 'interim_data.csv'),
        index=False,
    )
    with open(all_features.path, 'w') as features_file:
        features_file.write(json.dumps(final_df.columns.to_list()))
