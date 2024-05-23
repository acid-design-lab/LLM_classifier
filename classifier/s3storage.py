import os
from pathlib import Path
from uuid import uuid4
import requests
import magic
import boto3

class S3Storage:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
            endpoint_url=os.environ.get("S3_ENDPOINT_URL")
        )
        self.bucket = os.environ.get("S3_BUCKET")
    def store(self, filepath: str) -> str:
        filepath = Path(filepath)
        uuid = uuid4()
        res_filename = os.environ.get("S3_ENDPOINT_URL") + "/" + self.bucket + "/" + str(uuid) + filepath.suffix
        # requests.put(res_filename, filepath.open("rb"), headers={
        #     "Content-Type": magic.from_file(str(filepath)),
        #     "Authorization": self.__s3_access_key
        # })
        self.s3.upload_file(filepath, self.bucket, str(uuid) + filepath.suffix)
        return res_filename
