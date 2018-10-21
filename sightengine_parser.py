import json
import argparse
import glob
import os
import time

from sightengine.client import SightengineClient


class SightEngineParser():
    def load_apis(self):
        with open('api_keys.json') as f:
            self.api_keys = json.load(f)

    def initialize_client(self):
        self.load_apis()
        self.client = SightengineClient(self.api_keys['SIGHTENGINE_API_ID'], self.api_keys['SIGHTENGINE_API_KEY'])

    def fetch_results(self, filepath):
        try:
            return self.client.check('nudity', 'type', 'properties', 'wad', 'face').set_file(filepath)
        except:
            return None


if __name__ == "__main__":
    """
        python sightengine_parser.py --input_image_folder images/gambling --class_type gambling
        python sightengine_parser.py --input_image_folder images/drugs --class_type drugs
        python sightengine_parser.py --input_image_folder images/negative --class_type negative
        python sightengine_parser.py --input_image_folder images/nudity --class_type nudity
        python sightengine_parser.py --input_image_folder images/violence --class_type violence
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_folder", default="", help="")
    parser.add_argument("--class_type", default="", help="")
    args = parser.parse_args()

    print(args)

    with open('api_keys.json') as f:
        api_keys = json.load(f)

    app = SightEngineParser()
    app.initialize_client()

    # get the general model

    # output folder
    dirName = 'results_sightengine'

    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")

    all_violence_files = glob.glob('{}/*'.format(args.input_image_folder))
    print(all_violence_files)

    # result = predict_with_local_file('images/Man-holding-a-knife-via-Shutterstock.jpg')
    for fp in all_violence_files:
        # print(fp)

        file_name = fp.split('/')[-1]
        output_json_path = dirName + '/{}_image_results/{}.json'.format(args.class_type, os.path.basename(fp))

        if os.path.exists(output_json_path):
            print('Path exists: {}'.format(output_json_path))
            continue

        result_sightengine = app.fetch_results(os.path.abspath(fp))
        print('Outputting JSON to: {}'.format(output_json_path))
        with open(output_json_path, 'w') as fp:
            json.dump(result_sightengine, fp)
        # break
        time.sleep(60)
