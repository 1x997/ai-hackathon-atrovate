import json

from sightengine.client import SightengineClient


class SightEngineParser():
    def load_apis(self):
        with open('api_keys.json') as f:
            self.api_keys = json.load(f)
    def initialize_client(self):
        self.client = SightengineClient(self.api_keys['SIGHTENGINE_API_ID'], self.api_keys['SIGHTENGINE_API_KEY'])

    def fetch_results(self, filepath):
        try:
            return self.client.check('wad').set_url(filepath)
        except:
            return None

