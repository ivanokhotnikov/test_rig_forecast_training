# Training pipeline

The repo contains the source code and pipeline configurations to automate retraining of the test rig forecaster. The code uses Vertex AI Pipelines with Kubeflow SDK to orchestrate execution of the pipeline.

## Pipeline diagram

![Training pipeline](https://github.com/ivanokhotnikov/test_rig_forecast_training/blob/master/images/training_pipeline.png?raw=true)

## Pipeline steps

|Step   |Component          |Description                                                                                                                               |Inputs     |Outputs                           |Parameters    |Base image        |
|:---   |:---               |:---                                                                                                                                      |:---       |:---                              |:---          |:---              |
|1      |`read-raw-data`    |Reads raw data files from the GCS raw data bucket storage. Uploads the combined data frame to the interim data directory in the GCS bucket|           |`interim_data`<br/>`all_features` |              |`python:3.10-slim`|
|1a     |`importer`         |Imports interim features                                                                                                                  |           |`interim_features`                |`artifact_uri`|                  |
|2      |`build-features`   |Reads the interim data, builds features (float down casting, removes NaNs and the step zero data, adds the power and time features to the processed data), saves the processed data|`interim_features`<br/>`interim_data`|`processed_data`<br/>`processed_features`| |`python:3.10-slim`|
|3      |`split-data`       |Splits processed data into train and test data|`processed_data`|`train_data`<br/>`test_data`|`train_data_size`|`python:3.10-slim`|
|3a     |`import-forecast-features`|Imports forecast features||`forecast_features`|||
|4      |`train`            |Instantiates, trains the RNN model on the train dataset. Saves the trained scaler and the keras model to the metadata store, logs the training metrics and tensorboard event file|`train_data`|`scaler_model`<br/>`keras_model`<br/>`train_metrics`|`feature`<br/>`lookback`<br/>`lstm_units`<br/>`learning_rate`<br/>`epochs`<br/>`batch_size`<br/>`patience`|`tensorflow:latest-gpu`|
|5      |`evaluate`         |Evaluates the trained keras model, saves the evaluation metrics to the metadata store|`test_data`<br/>`scaler_model`<br/>`keras_model`<br/>|`eval_metrics`<br/>|`feature`<br/>`lookback`<br/>`lstm_units`<br/>`learning_rate`<br/>`epochs`<br/>`batch_size`<br/>`patience`|`tensorflow:latest-gpu`|
|5a     |`import-champion-metrics`|Imports champion metrics||`champion_metrics`|||
|6      |`compare-models`   |Compares evaluation metrics of the trained (challenger) model and the champion (the one in the model registry)|`eval_metrics`<br/>`champion_metrics`|`output`|`evaluation_metric`<br/>`absolute_difference`|`python:3.10-slim`|
|7      |`upload-model-to-registry`|Uploads the scaler and keras models to the models registry. Uploads the parameters and metrics of the model|`scaler_model`<br/>`keras_model`<br/>`eval_metrics`||`feature`<br/>`lookback`<br/>`lstm_units`<br/>`learning_rate`<br/>`epochs`<br/>`batch_size`<br/>`patience`|`tensorflow:latest-gpu`

## Directed acyclic graph diagram

![Training pipeline's DAG](https://github.com/ivanokhotnikov/test_rig_forecast_training/blob/master/images/training_dag.png?raw=true)
