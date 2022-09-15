import google.cloud.aiplatform as aip
import time

if __name__ == '__main__':
    aip.init(
        project='test-rig-349313',
        location='europe-west2',
    )
    for job in aip.CustomJob.list():
        job.delete()
        time.sleep(2)
    for art in aip.Artifact.list():
        art.delete()
        time.sleep(2)