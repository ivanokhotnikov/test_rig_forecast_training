import os

from kfp.v2.dsl import Artifact, Dataset, Input, Output, component


@component(
    base_image='python:3.10-slim',
    packages_to_install=['pandas'],
    output_component_file=os.path.join('configs', 'build_features.yaml'),
)
def build_features(
    raw_features: Input[Artifact],
    interim_data: Input[Dataset],
    processed_data: Output[Dataset],
) -> None:
    """
    Read the interim data, build features (float down casting, removes NaNs and the step zero data, calculates and adds to the processed data the power and time features), saves the processed data.

    Args:
        raw_features (Input[Artifact]): Raw features json artifact
        interim_data (Input[Dataset]): Interim dataset
        processed_data (Output[Dataset]): Processed dataset
    """
    import json
    import math
    import os

    import pandas as pd

    with open(raw_features.path, 'r') as features_file:
        raw_features_list = list(json.loads(features_file.read()))
    no_time_features = [
        f for f in raw_features_list if f not in ('TIME', ' DATE')
    ]
    df = pd.read_csv(
        interim_data.path + '.csv',
        usecols=raw_features_list,
        header=0,
        index_col=False,
        low_memory=False,
    )
    df[no_time_features] = df[no_time_features].apply(pd.to_numeric,
                                                      errors='coerce',
                                                      downcast='float')
    df.dropna(axis=0, inplace=True)
    df.drop(df[df['STEP'] == 0].index, axis=0).reset_index(drop=True)
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
    df['DURATION'] = pd.to_timedelta(range(len(df)), unit='s')
    df['RUNNING_SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(int)
    df['RUNNING_HOURS'] = (df['RUNNING_SECONDS'] / 3600).astype(float)
    df.columns = df.columns.str.lstrip()
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv(
        os.path.join('gcs', 'test_rig_data', 'processed',
                     'processed_data.csv'),
        index=False,
    )
    df.to_csv(
        processed_data.path + '.csv',
        index=False,
    )
