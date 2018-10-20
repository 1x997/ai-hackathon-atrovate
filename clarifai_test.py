import json
from pprint import pprint

from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

with open('api_keys.json', 'r') as f:
    api_keys = json.loads(f)

app = ClarifaiApp(api_key=api_keys['CLARIFAI_API_KEY'])

# get the general model
model = app.models.get("general-v1.3")

def predict_with_local_file(image_path):
    image = ClImage(file_obj=open(image_path, 'rb'))
    result = model.predict([image])
    return result

def predict_with_url(image_url):
    result = model.predict_by_url(url='image_url')
    return result

result = predict_with_local_file('images/Man-holding-a-knife-via-Shutterstock.jpg')

print('Model: ', model)
print('Output results: \n\n')
pprint(result)

with open('sample_clarifai_result.json', 'w') as fp:
    json.dump(result, fp)