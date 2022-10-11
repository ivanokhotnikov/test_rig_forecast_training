import os
from datetime import datetime

import google.cloud.aiplatform as aip
from kfp.v2 import compiler
from kfp.v2.dsl import Artifact, ParallelFor, importer, pipeline

from components import (build_features, evaluate, import_final_features,
                        read_raw_data, split_data, train)
from utils.constants import (PROJECT_ID, REGION, PIPELINES_BUCKET_URI,
                             TRAIN_GPU, TRAIN_NGPU, VCPU, MEMORY_LIMIT)


@pipeline(name='training-pipeline', pipeline_root=PIPELINES_BUCKET_URI)
def training_pipeline(
    train_data_size: float,
    lookback: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
) -> None:
    raw_features_import = importer(
        artifact_uri='gs://test_rig_data/features/raw_features.json',
        artifact_class=Artifact,
    )
    read_raw_data_task = read_raw_data()
    build_features_task = build_features(
        raw_features=raw_features_import.output,
        interim_data=read_raw_data_task.outputs['interim_data'],
    )
    split_data_task = split_data(
        train_data_size=train_data_size,
        processed_data=build_features_task.outputs['processed_data'])
    final_features_import = import_final_features()
    with ParallelFor(final_features_import.output) as feature:
        train_task = (train(
            feature=feature,
            lookback=lookback,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            train_data=split_data_task.outputs['train_data'],
        )
        .set_cpu_limit(VCPU)\
        .set_memory_limit(MEMORY_LIMIT)
        .add_node_selector_constraint('cloud.google.com/gke-accelerator','NVIDIA_TESLA_T4')\
        .set_gpu_limit(TRAIN_NGPU))
        evaluate_task = (evaluate(
            feature=feature,
            lookback=lookback,
            batch_size=batch_size,
            test_data=split_data_task.outputs['test_data'],
            scaler_model=train_task.outputs['scaler_model'],
            keras_model=train_task.outputs['keras_model'],
        )
        .set_cpu_limit(VCPU)\
        .set_memory_limit(MEMORY_LIMIT)
        .add_node_selector_constraint('cloud.google.com/gke-accelerator','NVIDIA_TESLA_T4')\
        .set_gpu_limit(TRAIN_NGPU))


if __name__ == '__main__':
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
        display_name='train_' + datetime.now().strftime('%Y%m%d%H%M%S'),
        pipeline_root=PIPELINES_BUCKET_URI,
        template_path=os.path.join('configs', 'training_pipeline.json'),
        parameter_values={
            'train_data_size': 0.8,
            'lookback': 120,
            'lstm_units': 5,
            'learning_rate': 0.01,
            'epochs': 30,
            'batch_size': 256,
            'patience': 5,
        },
    )
    job.run()
