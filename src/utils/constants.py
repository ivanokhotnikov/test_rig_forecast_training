import google.cloud.aiplatform as aip
import google.cloud.storage as storage

PROJECT_ID = 'test-rig-349313'
REGION = 'europe-west2'
PROJECT_NUMBER = 42869708044

DATA_BUCKET_NAME = 'test_rig_data'
DATA_BUCKET_URI = f'gs://{DATA_BUCKET_NAME}'
PIPELINES_BUCKET_NAME = 'test_rig_pipelines'
PIPELINES_BUCKET_URI = f'gs://{PIPELINES_BUCKET_NAME}'

TRAIN_GPU, TRAIN_NGPU = (aip.gapic.AcceleratorType.NVIDIA_TESLA_T4, 2)
TRAIN_CPU_PREFIX = 'tf-cpu.2-9'
TRAIN_GPU_PREFIX = 'tf-gpu.2-9'
TRAIN_CPU_IMAGE = f'{REGION.split("-")[0]}-docker.pkg.dev/vertex-ai/training/{TRAIN_CPU_PREFIX}:latest'
TRAIN_GPU_IMAGE = f'{REGION.split("-")[0]}-docker.pkg.dev/vertex-ai/training/{TRAIN_GPU_PREFIX}:latest'

MACHINE_TYPE = 'n1-standard'
VCPU = '8'
TRAIN_COMPUTE = MACHINE_TYPE + '-' + VCPU
MEMORY_LIMIT = '32G'
SLEEP_TIME = 1.05

VERTEX_REGIONS = {'europe-west1', 'europe-west2', 'europe-west4'}

STORAGE_CLIENT = storage.Client()
