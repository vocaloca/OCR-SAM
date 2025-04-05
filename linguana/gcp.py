import datetime
from google.cloud import storage
import os
from logging import getLogger
from typing import Union
from google.cloud.storage.blob import Blob

logger = getLogger(__file__)


def download_blob(bucket_name: str, source_blob_prefix: str, destination_dir: str, force: bool = False):
    """Downloads all blobs with a given prefix (folder) from the bucket, maintaining structure."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = storage_client.list_blobs(bucket_name, prefix=source_blob_prefix)
    for blob in blobs:
        # Compute relative path within the destination directory
        relative_path = blob.name[len(source_blob_prefix):].lstrip("/")
        destination_file_name = os.path.join(destination_dir, relative_path)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

        # Skip "directory" blobs (empty blobs used as folder markers)
        if not blob.name.endswith("/"):
            if force or not os.path.exists(destination_file_name):
                blob.download_to_filename(destination_file_name)
                print(f"✅ Downloaded: gs://{bucket_name}/{blob.name} -> {destination_file_name}")
            else:
                print(f"⚠️ Skipped (exists): {destination_file_name}")


def download_blob_from_gcs_simplified(image_path: str, bucket_name: str = "supertale-misc"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(image_path)
    data = blob.download_as_bytes()
    return data

def upload_to_gcs(bucket_name: str, local_file_path: str, destination_blob_name: str, is_public: bool = False):
    """Uploads a file to Google Cloud Storage and returns its signed url or a  public url."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload file
    blob.upload_from_filename(local_file_path)
    
    # Make file public
    if is_public:
        blob.make_public()
        url = blob.public_url
    else:
        url = generate_download_signed_url_v4(bucket_name, blob.name)
    
    return url

def generate_signed_url(bucket_name: str, blob: Union[Blob, str]) -> str:
    """Generates a signed URL for downloading a blob."""
    if isinstance(blob, str):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow GET requests using this URL.
        method="GET",
    )

    return url

def delete_from_gcs(bucket_name: str, file_path: str) -> None:
    """Deletes a file from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    try:
        blob.delete()
        print(f"File {file_path} deleted successfully from GCS.")
    except Exception as e:
        print(f"Error deleting file: {e}")


def generate_download_signed_url_v4(bucket_name: str, blob_name: str) -> str:
    """Generates a v4 signed URL for downloading a blob.

    Note that this method requires a service account key file. You can not use
    this if you are using Application Default Credentials from Google Compute
    Engine or from the Google Cloud SDK.
    """
    # bucket_name = 'your-bucket-name'
    # blob_name = 'your-object-name'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow GET requests using this URL.
        method="GET",
    )

    print("Generated GET signed URL:")
    print(url)
    print("You can use this URL with any user agent, for example:")
    print(f"curl '{url}'")
    return url


def get_secret(project_id, secret_id):
    from google.cloud import secretmanager

    try:
        client = secretmanager.SecretManagerServiceClient()
        name = client.secret_version_path(project_id, secret_id, "latest")
        response = client.access_secret_version(request={"name": name})
        openai_api_key = response.payload.data.decode("UTF-8")
        return openai_api_key
    except Exception as e:
        logger.error(f"get_secret error: {e}")
        raise


if __name__ == "__main__":
    # download_blob('linguana-models', 'image-eraser/', '/checkpoints/', force=False)
    print(generate_download_signed_url_v4("linguana-models", "image-eraser-for-captioning/erase_2.jpg"))
