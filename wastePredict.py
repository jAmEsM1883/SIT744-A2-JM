###############################################################################
# Name: SIT744 Assignment 2 - Waste Image Classification Prediction.
# File: wastePredict.py
# Build: 1.0
# Organisation: Deakin University - SIT744 - Deep Learning
# @author: James Martin
###############################################################################
# Purpose:  Predict whether the item(s) in an image are recyclable or not.
#
# Overview: Load a pre-trained TensorFlow model from Google Cloud Storage.
#           Load the user supplied image.
#           Convert the image to a suitable format.
#           Use the TensorFlow model to predict the expected target class.
#           Report the predicted target class to the user.
###############################################################################
#
# Some of the code in this file is based on the following pages:
# https://cloud.google.com/functions/docs/writing/http#multipart_data
# https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions
#
###############################################################################
#
# requirements.txt
#
# numpy
# Pillow
# tensorflow
# google-cloud-storage
# pathlib
#
###############################################################################

# Import standard Python libraries.
import os
import tempfile
import PIL.Image as Image
import pathlib

# Import Python ML/DL libraries.
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import GCP libraries.
from werkzeug.utils import secure_filename
from google.cloud import storage

# Use a global variable for the model to reduce continually reloading of the model.
model = None

def getFilePath(filename):
    """
    Gets the temporary location of the user supplied image in the GCP file system.

    Args:
        filename: Image filename

    Returns:
        Path to the file in GCP temporary storage.
    """
    fileName = secure_filename(filename)
    return os.path.join(tempfile.gettempdir(), fileName)

def processPredict(request):
    """
    Use a pre-trained TensorFlow model to predict the target class of a user supplied image.

    Args:
        request: Flask.Request object from the assigments HTML page in Cloud Storage.

    Returns:
        Predicted target class.
    """
    # Future - return a Response object using `make_response` <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.

    predictionResult = "Prediction Result - "
    predictionClasses = ['non-recyclable', 'recyclable']
    predictionDecision = "Error - Prediction Failure" # Default value
    imageShape = (180, 180)
#    downloadBucketFiles = ['task2-m1-20220515-202340/keras_metadata.pb', 'task2-m1-20220515-202340/saved_model.pb', 'task2-m1-20220515-202340/variables/variables.data-00000-of-00001', 'task2-m1-20220515-202340/variables/variables.index']
#    downloadLocalFiles = ['/tmp/keras_metadata.pb', '/tmp/saved_model.pb', '/tmp/variables/variables.data-00000-of-00001', '/tmp/variables/variables.index'] 

    # Connect to the Google Cloud Storage Bucket for this assignment and retrieve the model.
    bucketName = "sit744-bucket-jm"

    print('Getting files from Cloud Storage Bucket')
    storageClient = storage.Client()
    bucket = storageClient.bucket(bucketName)
    print('Connected to bucket ' + bucketName)
    blobs = storageClient.list_blobs(bucketName)
    tempDir = tempfile.gettempdir()
    for blob in blobs:
        print(f'{blob.name}')
        (top, tail) = os.path.split(blob.name)
        if len(top) != 0:
            try:
                newDir = os.path.join(tempDir, top)
                pathlib.Path(newDir).mkdir(parents = True, exist_ok = True)
            except Exception as e:
                print(f'Directory {newDir} already exists. ' + str(repr(e)))
            else:
                print(f'Created directory {newDir}')
        print(f'Downloading {blob.name}')
        currentBlob = bucket.blob(blob.name)
        try:
            fullFileName = os.path.join(tempDir, blob.name)
            currentBlob.download_to_filename(fullFileName)
        except Exception as e:
            print(f'EXC downloading blob failed {str(repr(e))} {fullFileName}')
        else:
            print(f'Downloaded {blob.name}')

    # Load the model
    model = "task2-m1-20220515-202340"
    fullModel = os.path.join(tempDir, model)
    try:
        model = tf.keras.models.load_model(fullModel)
    except Exception as e:
        print("Model loading failed " + str(repr(e))+ ". " + fullModel)
    else:
        print("Model loading succeeded. " + fullModel)

    # Non-file fields from the form aren't expected and will be ignored.
    # request.form.to_dict()

    # Process the uploaded file. It is expected that only a single file will be uploaded.
    files = request.files.to_dict()
    for fileName, file in files.items():
        # Save the uploaded file to temporary storage.
        file.save(getFilePath(fileName))
        print(f'Processing file: {fileName}')
        try:
            testImage = Image.open(getFilePath(fileName)).resize(imageShape)
            testImage = np.array(testImage)[np.newaxis, ...] # Convert image to a numpy array and put it within a batch.
            logits = model.predict(testImage)
            print(f"Prediction complete {logits}")
            prediction = tf.nn.sigmoid(logits[0]) # Perform sigmoid activation for binary classification.
            print(f"Activation complete {prediction}")
            predictedClass = int(np.rint(prediction)) # Round up or down to determine the predicted class.
            print(f"Class determined {predictedClass}")
            predictionDecision = predictionClasses[predictedClass]
            print(f'Image {fileName} logits {logits} prediction {prediction} predictedClass {predictedClass} className {classNames[int(predictedClass)]}')
        except Exception as e:
            print("Exception during prediction " + str(repr(e)))

    # Remove temporary files.
    for fileName in files:
        file_path = getFilePath(fileName)
        os.remove(file_path)

    return predictionResult + predictionDecision