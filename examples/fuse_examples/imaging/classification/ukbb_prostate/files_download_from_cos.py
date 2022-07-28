from botocore.client import Config, ClientError
import ibm_boto3
import os
import json
import time
import glob

from typing import List
from fuse.utils import NDict

def download_sample_files(sample_ids: List[str], mri_output_dir:str, cos_cfg:NDict):
    if not os.path.exists(mri_output_dir):
        os.makedirs(mri_output_dir)
        print("created", mri_output_dir)

    fields = cos_cfg["fields"]
    potentially_missing_files_set = set()

    for field in fields:
        potentially_requested_files_set = set([s.replace("*", str(field)) for s in sample_ids])
        existing_files_disk_set = set([os.path.basename(f) for f in glob.glob(os.path.join(mri_output_dir, f"*_{field}_*_0.zip"))])

        potentially_missing_files_set |= potentially_requested_files_set - existing_files_disk_set

    if len(potentially_missing_files_set)==0:
        print("No missing files.")
        return

    print(f"checking download of {len(potentially_missing_files_set)} potnetially missing files")

    mri_type_bucket_name = cos_cfg["bucket_name"]


    cos_credentials_file = cos_cfg["credentials_file"]

    with open(cos_credentials_file, 'r') as f:
        cos_credentials = json.load(f)
        print(cos_credentials)

    auth_endpoint = 'https://iam.bluemix.net/oidc/token'
    service_endpoint = 'https://s3-api.us-geo.objectstorage.softlayer.net'


    t0 = time.time()
    cosClient = ibm_boto3.client('s3',
                                 ibm_api_key_id=cos_credentials['apikey'],
                                 ibm_service_instance_id=cos_credentials['resource_instance_id'],
                                 ibm_auth_endpoint=auth_endpoint,
                                 config=Config(signature_version='oauth'),
                                 endpoint_url=service_endpoint)
    print(f"connected to COS in time={time.time() - t0:.2f} seconds")

    # List the objects in the bucket
    existing_files_cos = listFilesFromCOS(cosClient, mri_type_bucket_name)
    print(len(existing_files_cos))


    filenames_to_download = list(set(existing_files_cos).intersection(potentially_missing_files_set))

    downloadFilesFromCOS(cosClient, filenames_to_download, mri_type_bucket_name, destinationDirectory=mri_output_dir)


# Create method to get a complete list of the files in a bucket (each call is limited to 1000 files)
def listFilesFromCOS(cosClient, bucketName):
    # Initialize result
    existingFileSet = set()

    # Initialize the continuation token
    ContinuationToken = ''

    # Loop in case there are more than 1000 files
    while True:

        # Get the file list (include continuation token for paging)
        res = cosClient.list_objects_v2(Bucket=bucketName, ContinuationToken=ContinuationToken)

        # Put the files in the set
        if 'Contents' in res:
            for r in res['Contents']:
                existingFileSet.add(r['Key'])

        # Check if there are more files and grab the continuation token if true
        if res['IsTruncated'] == True:
            ContinuationToken = res['NextContinuationToken']
        else:
            break

    # Return the file set
    return list(existingFileSet)


def downloadFilesFromCOS(cosClient, fileList, myBucketName, destinationDirectory='.'):
    print("prepare to download", len(fileList))
    destFileList = []

    # Loop over the files
    n_download = 0
    n_exist = 0
    for sourceFileName in fileList:
        # Get the destination file name
        destFileName = os.path.join(destinationDirectory, os.path.basename(sourceFileName))
        if os.path.exists(destFileName):
            print("File already exists", destFileName, "==> skipping")
            n_exist += 1
            continue

        # Copy file from my bucket
        cosClient.download_file(Filename=destFileName, Bucket=myBucketName, Key=sourceFileName)

        print(n_download, sourceFileName, ' --> ', destFileName)
        n_download += 1

        destFileList.append(destFileName)

    print("Files:", n_download,"downloaded,",n_exist, "already existed")
    return destFileList

