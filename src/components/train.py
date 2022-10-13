from kfp.v2.dsl import component, Input, Output, Dataset, Metrics, Model


@component(
    base_image='tensorflow/tensorflow:latest-gpu',
    packages_to_install=[
        'pandas',
        'scikit-learn',
        'google-cloud-aiplatform',
        'protobuf==3.13.0',
    ],
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
    from datetime import datetime

    import google.cloud.aiplatform as aip
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')

    train_df = pd.read_csv(train_data.path + '.csv', index_col=False)
    train_data = train_df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaler_model.metadata['feature'] = feature
    joblib.dump(
        scaler,
        scaler_model.path + '.joblib',
    )
    joblib.dump(
        scaler,
        os.path.join('gcs', 'models_forecasting', f'{feature}.joblib'),
    )
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_train_data)):
        x_train.append(scaled_train_data[i - lookback:i])
        y_train.append(scaled_train_data[i])
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    forecaster = keras.models.Sequential(name=f'{feature}_forecaster')
    forecaster.add(
        keras.layers.LSTM(lstm_units,
                          input_shape=(x_train.shape[1], x_train.shape[2]),
                          return_sequences=False))
    forecaster.add(keras.layers.Dense(1))
    forecaster.compile(
        loss=keras.losses.mean_squared_error,
        metrics=keras.metrics.RootMeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate))
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
                                         'gcs',
                                         'test_rig_pipelines',
                                         'tb',
                                         feature,
                                         TIMESTAMP,
                                     ),
                                     histogram_freq=1,
                                     write_graph=True,
                                     write_images=True,
                                     update_freq='epoch',
                                 )
                             ])
    for k, v in history.history.items():
        history.history[k] = [float(vi) for vi in v]
        metrics.log_metric(k, history.history[k])
    metrics.metadata['feature'] = feature
    keras_model.metadata['feature'] = feature
    forecaster.save(keras_model.path + '.h5')
    forecaster.save(os.path.join('gcs', 'models_forecasting', f'{feature}.h5'))
    # aip.init(
    #     project=PROJECT_ID,
    #     location=REGION,
    # )
    # models = [
    #     model.display_name for model in aip.Model.list(
    #         project=PROJECT_ID,
    #         location=REGION,
    #     )
    # ]
    # if feature not in models:
    #     _ = aip.Model.upload(
    #         project=PROJECT_ID,
    #         location=REGION,
    #         display_name=feature,
    #         artifact_uri=f'{MODELS_BUCKET_URI}/{feature}',
    #         serving_container_image_uri=SERVING_IMAGE,
    #         is_default_version=True,
    #     )
    # else:
    #     for model in aip.Model.list(
    #             project=PROJECT_ID,
    #             location=REGION,
    #     ):
    #         if model.display_name == feature:
    #             _ = aip.Model.upload(
    #                 project=PROJECT_ID,
    #                 location=REGION,
    #                 parent_model=model.name,
    #                 display_name=feature,
    #                 artifact_uri=f'{MODELS_BUCKET_URI}/{feature}',
    #                 serving_container_image_uri=SERVING_IMAGE,
    #                 is_default_version=True,
    #             )
    #             break
