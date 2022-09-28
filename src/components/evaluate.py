import os

from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image='tensorflow/tensorflow:latest',
    packages_to_install=[
        'pandas',
        'scikit-learn',
        'google-cloud-aiplatform',
    ],
    output_component_file=os.path.join('configs', 'evaluate.yaml'),
)
def evaluate(
    feature: str,
    lookback: int,
    batch_size: int,
    test_data: Input[Dataset],
    scaler_model: Input[Model],
    keras_model: Input[Model],
    metrics: Output[Metrics],
) -> None:
    """Evaluates the trained keras model, saves the evaludation metrics to the metadata store

    Args:
        feature (str): Feature strin to train on
        lookback (int): Length of the lookback window
        batch_size (int): Batch size
        test_data (Input[Dataset]): Train dataset
        scaler_model (Input[Model]): Scaler model
        keras_model (Input[Model]): Keras model
        eval_metrics (Output[Metrics]): Metrics
    """
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from tensorflow import keras

    test_df = pd.read_csv(test_data.path + '.csv', index_col=False)
    test_data = test_df[feature].values.reshape(-1, 1)
    scaler = joblib.load(scaler_model.path + f'_{feature}.joblib')
    scaled_test = scaler.transform(test_data)
    x_test, y_test = [], []
    for i in range(lookback, len(scaled_test)):
        x_test.append(scaled_test[i - lookback:i])
        y_test.append(scaled_test[i])
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)
    forecaster = keras.models.load_model(keras_model.path + f'_{feature}.h5')
    results = forecaster.evaluate(x_test,
                                  y_test,
                                  verbose=1,
                                  batch_size=batch_size,
                                  return_dict=True)
    with open(metrics.path + f'_{feature}.json', 'w') as metrics_file:
        metrics_file.write(json.dumps(results))
    for k, v in results.items():
        metrics.log_metric(k, v)
    metrics.metadata['feature'] = feature
