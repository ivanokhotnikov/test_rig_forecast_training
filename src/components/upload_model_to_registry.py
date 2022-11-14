from kfp.v2.dsl import Input, Metrics, Model, component


@component(
    base_image='tensorflow/tensorflow:latest',
    packages_to_install=[
        'scikit-learn',
        'google-cloud-aiplatform',
        'protobuf==3.13.0',
    ],
)
def upload_model_to_registry(
    feature: str,
    train_data_size: float,
    lookback: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
    scaler_model: Input[Model],
    keras_model: Input[Model],
    metrics: Input[Metrics],
) -> None:
    """
    Uploads the scaler and keras models to the models registry. Uploads the parameters and metrics of the model.

    Args:
    feature (str): Feature string to train on
    train_data_size (str): Proportion of train/test split
    lookback (int): Length of the lookback window
    lstm_units (int): Number of the LSTM units in the RNN
    learning_rate (float): Initial learning rate
    epochs (int): Number of epochs to train
    batch_size (int): Batch size
    patience (int): Number of patient epochs before the callbacks activate
    train_data (Input[Dataset]): Train dataset
    scaler_model (Output[Model]): Scaler model
    keras_model (Output[Model]): Keras model
    metrics (Output[Metrics]): Evaluation metrics
    """
    import json
    import os

    import google.cloud.aiplatform as aip
    import joblib
    from tensorflow import keras
    PROJECT_ID = 'test-rig-349313'
    REGION = 'europe-west2'
    DEPLOY_IMAGE = 'europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-10'
    HPARAMS = {
        'train_data_size': train_data_size,
        'lookback': lookback,
        'lstm_units': lstm_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
    }

    scaler = joblib.load(scaler_model.path + '.joblib')
    joblib.dump(
        scaler,
        os.path.join('gcs', 'models_forecasting', f'{feature}.joblib'),
    )
    forecaster = keras.models.load_model(keras_model.path + '.h5')
    forecaster.save(os.path.join('gcs', 'models_forecasting', f'{feature}.h5'))
    with open(metrics.path + '.json', 'r') as pipeline_metrics_file:
        eval_metrics_dict = json.load(pipeline_metrics_file)
    with open(
            os.path.join('gcs', 'models_forecasting', f'{feature}.json'),
            'w',
    ) as registry_metrics_file:
        registry_metrics_file.write(json.dumps(eval_metrics_dict))
    with open(
            os.path.join('gcs', 'models_forecasting',
                         f'{feature}_params.json'),
            'w',
    ) as registry_params_file:
        registry_params_file.write(json.dumps(HPARAMS))
    aip.init(project=PROJECT_ID, location=REGION)
    models = [
        model.display_name for model in aip.Model.list(
            project=PROJECT_ID,
            location=REGION,
        )
    ]
    forecaster.save(
        os.path.join('gcs', 'models_forecasting', 'registry', feature))
    if feature not in models:
        model = aip.Model.upload(
            project=PROJECT_ID,
            location=REGION,
            display_name=feature,
            artifact_uri=os.path.join('gcs', 'models_forecasting', 'registry',
                                      feature),
            serving_container_image_uri=DEPLOY_IMAGE,
            is_default_version=True,
        )
    else:
        for model in aip.Model.list(project=PROJECT_ID, location=REGION):
            if model.display_name == feature:
                model = aip.Model.upload(
                    project=PROJECT_ID,
                    location=REGION,
                    parent_model=model.name,
                    display_name=feature,
                    artifact_uri=os.path.join('gcs', 'models_forecasting',
                                              'registry', feature),
                    serving_container_image_uri=DEPLOY_IMAGE,
                    is_default_version=True,
                )
                break
