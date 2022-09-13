import os
from datetime import datetime

import google.cloud.aiplatform as aip
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, importer, component, pipeline, ParallelFor)

from components import (build_features, read_raw_data, split_data, train,
                        evaluate, load_final_features)

PROJECT_ID = 'test-rig-349313'
REGION = 'europe-west2'
DATA_BUCKET = 'test_rig_data'
DATA_BUCKET_URI = f'gs://{DATA_BUCKET}'
PIPELINES_BUCKET = 'test_rig_pipelines'
PIPELINES_BUCKET_URI = f'gs://{PIPELINES_BUCKET}'

TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
DISPLAY_NAME = 'train_' + TIMESTAMP


@pipeline(name='training-pipeline', pipeline_root=PIPELINES_BUCKET_URI)
def training_pipeline(
    data_bucket: str,
    train_data_size: float,
    lookback: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
) -> None:
    raw_features_import = importer(
        artifact_uri='gs://test_rig_data/raw_features.json',
        artifact_class=Artifact,
        reimport=True,
    ).set_display_name('Load raw features')
    read_raw_data_task = read_raw_data(data_bucket_name=data_bucket)
    build_features_task = build_features(
        raw_features=raw_features_import.output,
        interim_data=read_raw_data_task.outputs['interim_data'],
    )
    split_data_task = split_data(
        train_data_size=train_data_size,
        processed_data=build_features_task.outputs['processed_data'])
    final_features_import = load_final_features(data_bucket_name=data_bucket)
    with ParallelFor(final_features_import.output) as feature:
        train_task = train(
            feature,
            lookback=lookback,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            train_data=split_data_task.outputs['train_data'],
        )
        evaluate_task = evaluate(
            feature,
            lookback=lookback,
            batch_size=batch_size,
            test_data=split_data_task.outputs['test_data'],
            scaler_model=train_task.outputs['scaler_model'],
            keras_model=train_task.outputs['keras_model'],
        )


aip.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=PIPELINES_BUCKET_URI,
)
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path=os.path.join('configs', 'training_pipeline.json'),
)
job = aip.PipelineJob(
    enable_caching=True,
    display_name=DISPLAY_NAME,
    pipeline_root=PIPELINES_BUCKET_URI,
    template_path=os.path.join('configs', 'training_pipeline.json'),
    parameter_values={
        'data_bucket': DATA_BUCKET,
        'train_data_size': 0.8,
        'lookback': 120,
        'lstm_units': 3,
        'learning_rate': 0.1,
        'epochs': 3,
        'batch_size': 256,
        'patience': 5
    },
)
job.run()
