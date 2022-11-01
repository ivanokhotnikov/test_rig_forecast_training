from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image='tensorflow/tensorflow:latest-gpu',
    packages_to_install=[
        'pandas',
        'scikit-learn',
        'google-cloud-aiplatform',
        'protobuf==3.13.0',
    ],
)
def evaluate(
    feature: str,
    lookback: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    test_data: Input[Dataset],
    scaler_model: Input[Model],
    keras_model: Input[Model],
    eval_metrics: Output[Metrics],
) -> None:
    """Evaluates the trained keras model, saves the evaludation metrics to the metadata store

    Args:
        feature (str): Feature string to train on
        lookback (int): Length of the lookback window
        lstm_units (int): Number of the LSTM units in the RNN
        learning_rate (float): Initial learning rate
        epochs (int): Number of epochs to train
        batch_size (int): Batch size
        patience (int): Number of patient epochs before the callbacks activate
        test_data (Input[Dataset]): Train dataset
        scaler_model (Input[Model]): Scaler model
        keras_model (Input[Model]): Keras model
        eval_metrics (Output[Metrics]): Metrics
    """
    import json
    from datetime import datetime

    import google.cloud.aiplatform as aip
    import joblib
    import numpy as np
    import pandas as pd
    from tensorflow import keras

    PROJECT_ID = 'test-rig-349313'
    REGION = 'europe-west2'
    EXP_NAME = feature.lower().replace('_', '-')
    TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
    EVAL_TIMESTAMP = datetime.now().strftime('%H:%M:%S %a %d %b %Y')
    HPARAMS = {
        'lookback': lookback,
        'lstm_units': lstm_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
    }
    aip.init(
        experiment=EXP_NAME,
        project=PROJECT_ID,
        location=REGION,
    )
    aip.start_run(run=TIMESTAMP)
    test_df = pd.read_csv(test_data.path + '.csv', index_col=False)
    test_data = test_df[feature].values.reshape(-1, 1)
    scaler = joblib.load(scaler_model.path + '.joblib')
    scaled_test = scaler.transform(test_data)
    x_test, y_test = [], []
    for i in range(lookback, len(scaled_test)):
        x_test.append(scaled_test[i - lookback:i])
        y_test.append(scaled_test[i])
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)
    forecaster = keras.models.load_model(keras_model.path + '.h5')
    results = forecaster.evaluate(x_test,
                                  y_test,
                                  verbose=1,
                                  batch_size=batch_size,
                                  return_dict=True)
    results['evaluation_timestamp'] = EVAL_TIMESTAMP
    with open(eval_metrics.path + '.json', 'w') as metrics_file:
        metrics_file.write(json.dumps(results))
    for k, v in results.items():
        eval_metrics.log_metric(k, v)
        aip.log_metrics({k: v})
    for k, v in HPARAMS.items():
        aip.log_params({k: v})
    eval_metrics.metadata['feature'] = feature
    aip.end_run()
