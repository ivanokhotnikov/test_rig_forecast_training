import argparse
import os
from datetime import datetime

import google.cloud.aiplatform as aip
from kfp.v2 import compiler
from kfp.v2.dsl import Artifact, Condition, ParallelFor, importer, pipeline

from components import (build_features, compare_models, evaluate,
                        import_champion_metrics, import_forecast_features,
                        read_raw_data, split_data, train,
                        upload_model_to_registry)
from utils.constants import (MEMORY_LIMIT, PIPELINES_BUCKET_URI, PROJECT_ID,
                             REGION, TRAIN_NGPU, VCPU)


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
    interim_features_import = importer(
        artifact_uri='gs://test_rig_features/interim_features.json',
        artifact_class=Artifact,
    ).set_display_name('import interim features')
    read_raw_data_task = read_raw_data()
    build_features_task = build_features(
        interim_features=interim_features_import.output,
        interim_data=read_raw_data_task.outputs['interim_data'],
    )
    split_data_task = split_data(
        train_data_size=train_data_size,
        processed_data=build_features_task.outputs['processed_data'])
    forecast_features_import = import_forecast_features()
    with ParallelFor(forecast_features_import.output) as feature:
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
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            test_data=split_data_task.outputs['test_data'],
            scaler_model=train_task.outputs['scaler_model'],
            keras_model=train_task.outputs['keras_model'],
        )
        .set_cpu_limit(VCPU)\
        .set_memory_limit(MEMORY_LIMIT)
        .add_node_selector_constraint('cloud.google.com/gke-accelerator','NVIDIA_TESLA_T4')\
        .set_gpu_limit(TRAIN_NGPU))
        import_champion_metrics_task = import_champion_metrics(feature=feature)
        compare_task = compare_models(
            challenger_metrics=evaluate_task.outputs['eval_metrics'],
            champion_metrics=import_champion_metrics_task.
            outputs['champion_metrics'],
        )
        with Condition(compare_task.output == 'true', name='chall better'):
            upload_model_to_registry(
                feature=feature,
                scaler_model=train_task.outputs['scaler_model'],
                keras_model=train_task.outputs['keras_model'],
                metrics=evaluate_task.outputs['eval_metrics'],
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--compile_only', action='store_true')
    args = parser.parse_args()
    params = {
        'train_data_size': 0.8,
        'lookback': 120,
        'lstm_units': 1,
        'learning_rate': 1,
        'epochs': 4,
        'batch_size': 256,
        'patience': 30
    } if args.dry_run else {
        'train_data_size': 0.75,
        'lookback': 120,
        'lstm_units': 20,
        'learning_rate': 0.01,
        'epochs': 220,
        'batch_size': 256,
        'patience': 25
    }
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=os.path.join('configs', 'training_pipeline.json'),
    )
    if not args.compile_only:
        aip.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=PIPELINES_BUCKET_URI,
        )
        job = aip.PipelineJob(
            enable_caching=True,
            display_name='train_' + datetime.now().strftime('%Y%m%d%H%M%S'),
            pipeline_root=PIPELINES_BUCKET_URI,
            template_path=os.path.join('configs', 'training_pipeline.json'),
            parameter_values=params,
        )
        job.submit()
