"""
This file contains the three essential lambda functions for the ML pipeline. 

The first one is used to extract the image data and then serialize the data from S3.
The second one accepts the serialized data and then runs the inference.
The third on accepts the inference results and then uploads the results to S3.


Note: The second lambda function need import 'sagemaker' library, so you need upload the lambda from a zip.

Reference: https://docs.aws.amazon.com/lambda/latest/dg/python-package.html

Article in this link may not work --- https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/




Lambda first test function:
{
  "image_data": "",
  "s3_bucket": "your-sagemaker-role",
  "s3_key": "test/bicycle_s_000513.png"
}


Lambda second test function:
{
  "statusCode": 200,
  "body": {
    "image_data": "base64_image_data", (for instance, iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9y)
    "s3_bucket": "your-sagemaker-role",
    "s3_key": "test/bicycle_s_000513.png",
    "inferences": []
  }
}

Lambda third test function:
{
  "statusCode": 200,
  "body": "{\"statusCode\": 200, \"body\": {\"image_data\": \"base64_image_data", \"s3_bucket\": \"your_sagemaker_role\", \"s3_key\": \"test/bicycle_s_000513.png\", \"inferences\": []}, \"inferences\": \"[0.9703435301780701, 0.029656505212187767]\"}"
}
"""
# ---------------------------------------------
# --------------- The first one ---------------
# ---------------------------------------------
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    bucket = event['s3_bucket']
    key = event['s3_key']
    
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# ---------------------------------------------
# -------------- The second one ---------------
# ---------------------------------------------
import json
import sagemaker
import base64
import os
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2022-03-07-21-31-43-391'

def lambda_handler(event, context):

    # Decode the image data
    img = event['body']['image_data']
    image = base64.b64decode(img)

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)
    print(inferences)
    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }



# --------------------------------------------
# -------------- The third one ---------------
# --------------------------------------------
import json


THRESHOLD = 0.88

class Threshold_Error(Exception):
    pass

def lambda_handler(event, context):
    body = event["body"]
    data = json.loads(body)
    
    # Grab the inferences from the event
    inferences = data["inferences"][1:-1]
    inferences = inferences.split(',')
    inferences = [float(i) for i in inferences]
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) >= THRESHOLD
    

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        print("THRESHOLD_CONFIDENCE_MET")
        pass
    else:
        raise Threshold_Error("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }