import json

from sightengine.client import SightengineClient


with open('api_keys.json') as f:
    api_keys = json.load(f)

client = SightengineClient(api_keys['SIGHTENGINE_API_ID'], api_keys['SIGHTENGINE_API_KEY'])

output = client.check('wad').set_url('https://s3-eu-west-1.amazonaws.com/crimedetection/52612343-a-man-holding-knife-crime.jpg')

print(output)