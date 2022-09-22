import os

from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image='tensorflow/tensorflow:latest',
    packages_to_install=[
        'pandas',
        'scikit-learn',
        'google-cloud-aiplatform',
    ],
    output_component_file=os.path.join('configs', 'train.yaml'),
)
def train(
    feature: str,
    lookback: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    train_data: Input[Dataset],
    scaler_model: Output[Model],
    keras_model: Output[Model],
    metrics: Output[Metrics],
) -> None:
    """Instantiates, trains the RNN model on the train dataset. Saves the trained scaler and the keras model to the metadata store, saves the evaluation metrics file as well.

    Args:
        feature (str): Feature string to train on
        lookback (int): Length of the lookback window
        lstm_units (int): Number of the LSTM units in the RNN
        learning_rate (float): Initial learning rate
        epochs (int): Number of epochs to train
        batch_size (int): Batch size
        patience (int): Number of patient epochs before the callbacks activate
        train_data (Input[Dataset]): Train dataset
        scaler_model (Output[Model]): Scaler model
        keras_model (Output[Model]): Keras model
        metrics (Output[Metrics]): Metrics
    """
    import os
    import json

    import google.cloud.aiplatform as aip
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    PROJECT_ID = 'test-rig-349313'
    REGION = 'europe-west2'
    TRAIN_GPU, TRAIN_NGPU = (aip.gapic.AcceleratorType.NVIDIA_TESLA_T4, 1)
    TRAIN_VERSION = 'tf-gpu.2-9'
    TRAIN_IMAGE = f'{REGION.split("-")[0]}-docker.pkg.dev/vertex-ai/training/{TRAIN_VERSION}:latest'
    PIPELINES_BUCKET_NAME = 'test_rig_pipelines'
    PIPELINES_BUCKET_URI = f'gs://{PIPELINES_BUCKET_NAME}'

    train_df = pd.read_csv(train_data.path + '.csv', index_col=False)
    train_data = train_df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaler_model.metadata['feature'] = feature
    joblib.dump(
        scaler,
        scaler_model.path + f'_{feature}.joblib',
    )
    joblib.dump(
        scaler,
        os.path.join('gcs', 'models_forecasting', f'{feature}.joblib'),
    )
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_train)):
        x_train.append(scaled_train[i - lookback:i])
        y_train.append(scaled_train[i])
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    forecaster = keras.models.Sequential()
    forecaster.add(
        keras.layers.LSTM(lstm_units,
                          input_shape=(x_train.shape[1], x_train.shape[2]),
                          return_sequences=False))
    forecaster.add(keras.layers.Dense(1))
    forecaster.compile(
        loss=keras.losses.mean_squared_error,
        metrics=keras.metrics.RootMeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate))
    aip.init(
        experiment=feature.lower().replace('_', '-'),
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=PIPELINES_BUCKET_URI,
    )
    aip.start_run(run=feature.lower().replace('_', '-'), )
    history = forecaster.fit(x_train,
                             y_train,
                             shuffle=False,
                             epochs=epochs,
                             batch_size=batch_size,
                             validation_split=0.2,
                             verbose=1,
                             callbacks=[
                                 keras.callbacks.EarlyStopping(
                                     patience=patience,
                                     monitor='val_loss',
                                     mode='min',
                                     verbose=1,
                                     restore_best_weights=True,
                                 ),
                                 keras.callbacks.ReduceLROnPlateau(
                                     monitor='val_loss',
                                     factor=0.75,
                                     patience=patience // 2,
                                     verbose=1,
                                     mode='min',
                                 ),
                                 keras.callbacks.TensorBoard(
                                     log_dir=os.path.join(
                                         'gcs', 'test_rig_pipelines',
                                         f'{feature}'),
                                     histogram_freq=1,
                                     write_graph=True,
                                     write_images=True,
                                     update_freq='epoch',
                                 )
                             ])
    with open(metrics.path + f'_{feature}.json', 'w') as metrics_file:
        for k, v in history.history.items():
            history.history[k] = [float(vi) for vi in v]
            metrics.log_metric(k, history.history[k])
        metrics.log_metric('feature', feature)
        metrics_file.write(json.dumps(history.history))
    keras_model.metadata['feature'] = feature
    forecaster.save(keras_model.path + f'_{feature}.h5')
    forecaster.save(os.path.join('gcs', 'models_forecasting', f'{feature}.h5'))
    aip.end_run()