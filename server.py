from flask import Flask
app = Flask(__name__)

import pickle
from pprint import pprint
import json

from flask import jsonify
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

input_json = {"imagename": "12738503-d515-11e8-8fff-fd27fd2770eb", 
              "sightengine": 
                    {"status": "failure", 
                    "request": {"id": "req_3YDUXdWaHw5IrhpZ9pPUU", 
                    "timestamp": 1540114715.8631, "operations": 0}, 
                    "error": {"type": "usage_limit", "code": 32, "message": "Daily usage limit reached"}}, 
                    "azure": {"categories": [{"name": "people_portrait", "score": 0.796875, "detail": {"celebrities": []}}], 
                        "color": {"dominantColorForeground": "Grey", "dominantColorBackground": "Grey", "dominantColors": ["Grey", "White"],
                        "accentColor": "694638", "isBwImg": False}, 
                        "description": {"tags": ["person", "indoor", "man", "wearing", "glasses", "holding", "shirt", "sitting", "kitchen", "camera", "standing", "green", "young", "computer", "laptop", "smiling", "room", "table", "hat", "food", "white"], 
                        "captions": [{"text": "a man wearing glasses and smiling at the camera", "confidence": 0.923427805979976}]}, 
                        "requestId": "ab568467-0635-4698-9568-5e2a52ceed70", 
                        "metadata": {"width": 640, "height": 360, "format": "Jpeg"}}}



#with open(json_fp, 'r') as fp:
#        result = json.load(fp)
#

def load_model(path_to_model_pickle):
    with open(path_to_model_pickle, 'rb') as input:
        model_obj = pickle.load(input)

    return model_obj

def get_transformed_data(input_json):
    #print(input_json['sightengine'])

    print()

    #print(input_json['sightengine']['azure'])

    print()

    pprint(input_json)

    print([x['text'] for x in input_json['azure']['description']['captions']])
    print([x for x in input_json['azure']['description']['tags']])

    all_words = [x['text'] for x in input_json['azure']['description']['captions']]
    all_words = all_words[0].split()
    all_words.extend([x for x in input_json['azure']['description']['tags']])

    all_words = [x for x in all_words if x in vocab]

    X = vectorizer.transform(all_words)

    return X

@app.route("/predict")
def predict():
    X = get_transformed_data(input_json)

    import pdb;pdb.set_trace() 

    probas = clf.predict_proba(np.zeros((1, len(vocab))))
    return jsonify(list(probas))

    #return json.dumps(input_json)

if __name__ == '__main__':
    model_obj = load_model('latest_model_obj.pkl')

    #predict(input_json)

    vocab = model_obj['vocabulary']
    print('Vocab length: {}'.format(len(vocab)))

    clf = model_obj['clf']

    global vectorizer

    vectorizer = CountVectorizer(vocabulary=vocab)

    

    #import pdb;pdb.set_trace()

    app.run()