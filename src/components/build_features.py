from kfp.v2.dsl import Artifact, Dataset, Input, Output, component


@component(
    base_image='python:3.10-slim',
    packages_to_install=['pandas'],
)
def build_features(
    interim_features: Input[Artifact],
    interim_data: Input[Dataset],
    processed_data: Output[Dataset],
    processed_features: Output[Artifact],
) -> None:
    """
    Read the interim data, build features (float down casting, removes NaNs and the step zero data, calculates and adds to the processed data the power and time features), saves the processed data.

    Args:
        processed_features (Input[Artifact]): Raw features json artifact
        interim_data (Input[Dataset]): Interim dataset
        processed_data (Output[Dataset]): Processed dataset
        processed_features (Output[Artifact]): Processed features
    """
    import json
    import logging
    import math
    import os

    import pandas as pd

    with open(interim_features.path, 'r') as features_file:
        interim_features_list = list(json.loads(features_file.read()))
    no_time_features = [
        f for f in interim_features_list if f not in ('TIME', ' DATE', 'DATE')
    ]
    df = pd.read_csv(
        interim_data.path + '.csv',
        usecols=interim_features_list,
        header=0,
        index_col=False,
        low_memory=False,
    )
    logging.info(f'Interim dataframe was read')
    df[no_time_features] = df[no_time_features].apply(pd.to_numeric,
                                                      errors='coerce',
                                                      downcast='float')
    df.dropna(
        axis=0,
        inplace=True,
        subset=no_time_features,
    )
    df.drop(columns='DATE', inplace=True, errors='ignore')
    df.drop(columns=' DATE', inplace=True, errors='ignore')
    df.drop(columns='DURATION', inplace=True, errors='ignore')
    logging.info(f'NAs and date columns droped')
    df = df.drop(df[df['STEP'] == 0].index, axis=0).reset_index(drop=True)
    logging.info(f'Step zero removed')
    df['DRIVE_POWER'] = (df['M1 SPEED'] * df['M1 TORQUE'] * math.pi / 30 /
                         1e3).astype(float)
    df['LOAD_POWER'] = abs(df['D1 RPM'] * df['D1 TORQUE'] * math.pi / 30 /
                           1e3).astype(float)
    df['CHARGE_MECH_POWER'] = (df['M2 RPM'] * df['M2 Torque'] * math.pi / 30 /
                               1e3).astype(float)
    df['CHARGE_HYD_POWER'] = (df['CHARGE PT'] * 1e5 * df['CHARGE FLOW'] *
                              1e-3 / 60 / 1e3).astype(float)
    df['SERVO_MECH_POWER'] = (df['M3 RPM'] * df['M3 Torque'] * math.pi / 30 /
                              1e3).astype(float)
    df['SERVO_HYD_POWER'] = (df['Servo PT'] * 1e5 * df['SERVO FLOW'] * 1e-3 /
                             60 / 1e3).astype(float)
    df['SCAVENGE_POWER'] = (df['M5 RPM'] * df['M5 Torque'] * math.pi / 30 /
                            1e3).astype(float)
    df['MAIN_COOLER_POWER'] = (df['M6 RPM'] * df['M6 Torque'] * math.pi / 30 /
                               1e3).astype(float)
    df['GEARBOX_COOLER_POWER'] = (df['M7 RPM'] * df['M7 Torque'] * math.pi /
                                  30 / 1e3).astype(float)
    logging.info(f'Power features added')
    df['RUNNING_SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(int)
    df['RUNNING_HOURS'] = (df['RUNNING_SECONDS'] / 3600).astype(float)
    logging.info(f'Time features added')
    df.columns = df.columns.str.lstrip()
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv(
        os.path.join('gcs', 'test_rig_processed_data', 'processed_data.csv'),
        index=False,
    )
    logging.info(f'Processed dataframe uploaded to processed data storage')
    df.to_csv(
        processed_data.path + '.csv',
        index=False,
    )
    logging.info(f'Processed dataframe uploaded to metadata store')
    with open(processed_features.path + '.json', 'w') as features_file:
        json.dump(df.columns.to_list(), features_file)
    logging.info('Processed features uploaded to the pipeline metadata store')
    with open(
            os.path.join('gcs', 'test_rig_features',
                         'processed_features.json'), 'w') as features_file:
        json.dump(df.columns.to_list(), features_file)
    logging.info('Processed features uploaded to the featues store')
