import argparse
import json
from pprint import pprint

from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

import glob

def predict_with_local_file(image_path):
    image = ClImage(file_obj=open(image_path, 'rb'))
    result = model.predict([image])
    return result

def predict_with_url(image_url):
    result = model.predict_by_url(url='image_url')
    return result

if __name__ == "__main__":
    """
        python clarifai_run_on_folder.py --input_image_folder images/gambling --class_type gambling
        python clarifai_run_on_folder.py --input_image_folder images/drugs --class_type drugs
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_folder", default="", help="")
    parser.add_argument("--class_type", default="", help="")
    args = parser.parse_args()

    print(args)

    with open('api_keys.json') as f:
        api_keys = json.load(f)

    app = ClarifaiApp(api_key=api_keys['CLARIFAI_API_KEY'])

    # get the general model
    model = app.models.get("general-v1.3")

    all_violence_files = glob.glob('{}/*'.format(args.input_image_folder))
    print(all_violence_files)

    #result = predict_with_local_file('images/Man-holding-a-knife-via-Shutterstock.jpg')
    for fp in all_violence_files:
        file_name = fp.split('\\')[1]
        output_json_path = 'results/{}_image_results/{}.json'.format(args.class_type, file_name)
        
        result = predict_with_local_file(fp)

        #print('Model: ', model)
        #print('Output results: \n\n')
        #pprint(result)

        print('Outputting JSON to: {}'.format(output_json_path))
        with open(output_json_path, 'w') as fp:
            json.dump(result, fp)