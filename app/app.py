import sys
import json
import typing
import boto3
import tensorflow as tf
import os
from model import MyModel
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


s3 = boto3.client('s3')

BUCKET = 'melanoma-classification'
DESTINATION_FOLDER = '/tmp/'
IMAGE_SIZE = (224, 224)

cached_model: MyModel = None


def download_load_and_delete_image(image_key: str):

    # Download
    destination = os.path.join(DESTINATION_FOLDER, image_key.split('/')[-1])
    with open(destination, 'wb') as f:
        s3.download_fileobj(BUCKET, image_key, f)

    # Load
    image = tf.image.decode_jpeg(tf.io.read_file(destination))

    # Delete
    os.remove(destination)

    return image


def print_credentials():
    session = boto3.Session()
    credentials = session.get_credentials()

    # Credentials are refreshable, so accessing your access key / secret key
    # separately can lead to a race condition. Use this to get an actual matched
    # set.
    credentials = credentials.get_frozen_credentials()
    access_key = credentials.access_key
    secret_key = credentials.secret_key
    print("credentials", access_key, secret_key)


def _handler(event: typing.Dict, context):
    global cached_model

    logging.debug("\nEvent:", str(event), '\n')

    image_key = event['queryStringParameters']['image_key']
    age = float(event['queryStringParameters']['age'])
    sex = event['queryStringParameters']['sex']
    location = event['queryStringParameters']['location']

    image_tensor = download_load_and_delete_image(image_key)

    if cached_model is None:
        logging.info("Model no in cache. Loading...")
        cached_model = MyModel.create_standard_version("./weights/")
        logging.info("Model loaded.")

    prediction = cached_model.predict_single_image(
        image=image_tensor, sex=sex, age=age, location=location)

    return prediction


def handler(event: typing.Dict, context):
    try:
        # print_credentials()
        pred = _handler(event, context)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(
                {
                    "predicted_label": pred,
                }
            )
        }

    except Exception as e:
        logging.error(f"Event: {str(event)} - error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },

            'body': json.dumps(
                {
                    "error": str(e),
                    "event": str(event)
                }
            )
        }


if __name__ == "__main__":
    event = {
        "queryStringParameters": {
            "image_key": "input_images/test.jpg",
            "age": "45",
            "sex": "male",
            "location": "torso"
        }
    }

    handler(event, None)
