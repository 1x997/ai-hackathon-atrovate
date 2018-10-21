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
        self.client = SightengineClient(self.api_keys['SIGHTENGINE_API_ID'], self.api_keys['SIGHTENGINE_API_KEY'])

    def fetch_results(self, filepath):
        try:
            return self.client.check('wad').set_url(filepath)
        except:
            return None


if __name__ == "__main__":
    """
        python clarifai_run_on_folder.py --input_image_folder images/gambling --class_type gambling
        python clarifai_run_on_folder.py --input_image_folder images/drugs --class_type drugs
        python clarifai_run_on_folder.py --input_image_folder images/negative --class_type negative
        python clarifai_run_on_folder.py --input_image_folder images/nudity --class_type nudity
        python clarifai_run_on_folder.py --input_image_folder images/violence --class_type violence
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
    else:
        print("Directory ", dirName, " already exists")

    all_violence_files = glob.glob('{}/*'.format(args.input_image_folder))
    print(all_violence_files)

    # result = predict_with_local_file('images/Man-holding-a-knife-via-Shutterstock.jpg')
    for fp in all_violence_files:
        file_name = fp.split('\\')[1]
        output_json_path = dirName + '/{}_image_results/{}.json'.format(args.class_type, file_name)

        if os.path.exists(output_json_path):
            print('Path exists: {}'.format(output_json_path))
            continue

        result_sightengine = app.fetch_results()
        # azure

        # print('Model: ', model)
        # print('Output results: \n\n')
        # pprint(result)

        print('Outputting JSON to: {}'.format(output_json_path))
        with open(output_json_path, 'w') as fp:
            json.dump(result_sightengine, fp)

        time.sleep(60)
