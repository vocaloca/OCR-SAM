from google.cloud import storage
from google.cloud import pubsub_v1
import json

def list_files(bucket_name):
    """List all files in GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    """ Get all files from dubbiing/x/asdf/ """
    blobs = bucket.list_blobs(prefix='dubbing/30065014552/full/eraser-test/output')
    """ continue if the index of the filename such as output_frame_0001.jpg is less than 100"""
    blobs = [blob for blob in blobs if int(blob.name.split('/')[-1].split('_')[-1].split('.')[0]) > 300]
    
    file_names = [blob.name for blob in blobs]
    return file_names

def publish_message(project_id, topic_id, message):
    """Publish a message to a Pub/Sub topic."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    message_json = json.dumps(message)
    message_bytes = message_json.encode('utf-8')
    
    future = publisher.publish(topic_path, data=message_bytes)
    print(f'Published message ID: {future.result()}')

def main(bucket_name, project_id, topic_id):
    file_names = list_files(bucket_name)
    
    for file_name in file_names:
        print(f"working on {file_name}")
        message = {'image_path': file_name}
        publish_message(project_id, topic_id, message)

if __name__ == "__main__":
    bucket_name = "supertale-misc"
    project_id = "supertale-main"
    topic_id = "image-eraser"
    
    main(bucket_name, project_id, topic_id)
