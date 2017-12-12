'''
Simple code for prediction using Clarifai API
'''
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import ClarifaiApp
import json

app = ClarifaiApp(api_key='xxxxx')

# get the general model
model = app.models.get("general-v1.3")

# predict with the model
predictions=model.predict_by_url(url='http://filecremers3.informatik.tu-muenchen.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk_with_person-rgb.png')
for i in range(10):
    print predictions["outputs"][0]["data"]["concepts"][i]["name"]
    print predictions["outputs"][0]["data"]["concepts"][i]["value"]
