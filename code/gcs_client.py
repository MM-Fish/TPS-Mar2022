import os
from google.cloud import storage
import glob

BUCKET_NAME = 'kaggleops-bucket-msm'
BLOB_NAME = 'models'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../gcs-key.json'

class StorageClient():
  def __init__():
    '''
    '''
  def upload_gcs_from_directory(bucket: storage.bucket.Bucket, directory_path: str, blob_name: str, root_position=2):
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    for local_file in rel_paths:
      remote_path = f'{blob_name}/{"/".join(local_file.split(os.sep)[root_position:])}'
      if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)

  def upload_gcs_from_directory(bucket: storage.bucket.Bucket, directory_path: str, blob_name: str, root_position=2):
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    for local_file in rel_paths:
      remote_path = f'{blob_name}/{"/".join(local_file.split(os.sep)[root_position:])}'
      if os.path.isfile(local_file):
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)
