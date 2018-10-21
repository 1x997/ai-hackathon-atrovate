import pickle
from pprint import pprint
import json
import base64

from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from flask import jsonify, request
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   

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

def predict_with_local_file(image_path):
    image = ClImage(file_obj=open(image_path, 'rb'))
    result = model.predict([image])
    return result

def load_model(path_to_model_pickle):
    with open(path_to_model_pickle, 'rb') as input:
        model_obj = pickle.load(input)

    return model_obj

def get_transformed_data(result):
    #print(input_json['sightengine'])
    '''print()
    #print(input_json['sightengine']['azure'])
    print()
    pprint(input_json)

    print([x['text'] for x in input_json['azure']['description']['captions']])
    print([x for x in input_json['azure']['description']['tags']])

    all_words = [x['text'] for x in input_json['azure']['description']['captions']]
    all_words = all_words[0].split()
    all_words.extend([x for x in input_json['azure']['description']['tags']])

    all_words = [x for x in all_words if x in vocab]

    X = vectorizer.transform(all_words)'''

    all_concepts = result['outputs'][0]['data']['concepts']
    #print(all_concepts)
    #print('Number of concepts found: {}'.format(len(all_concepts)))
    concept_names_found = [x['name'] for x in all_concepts]
    names_and_scores = [{x['name'], x['value']} for x in all_concepts]
    sentence_for_result = ' '.join(concept_names_found)

    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform([sentence_for_result])
    #print(vectorizer.get_feature_names())
    X_full = X.toarray()
    print(concept_names_found)
    print('Sum of bag of words: {}'.format(X_full.sum()))

    return X_full

def get_probas_from_clfs(X, one_clf=False):
    which_proba = 0
    if one_clf:
        probas = clf.predict_proba(X)
        return list(probas[0])
    else:
        proba_violence = clf_violence.predict_proba(X)
        proba_gambling = clf_gambling.predict_proba(X)
        proba_drugs = clf_drugs.predict_proba(X)
        proba_nudity = clf_nudity.predict_proba(X)
        proba_negative = clf_negative.predict_proba(X)

        return [proba_violence[0][which_proba], proba_gambling[0][which_proba], proba_drugs[0][which_proba], 
                proba_nudity[0][which_proba], proba_negative[0][which_proba]]

@app.route("/predict", methods=['GET', 'OPTION', 'POST'])
def predict():
    #import pdb;pdb.set_trace()
    if request.method == 'POST':
        pass
        #print(request.data)
    else:
        return ''#jsonify({'No results'})

    global count
    if count < 5:
        img_str = json.loads(request.data)['image']
        imgdata = base64.b64decode(img_str)
        filename = 'current_image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgdata)

        result = predict_with_local_file(filename)

        X = get_transformed_data(result)
        #probas = clf.predict_proba(np.zeros((1, len(vocab))))
        #probas = get_probas_from_clfs(X, one_clf=False)
        probas = get_probas_from_clfs(X, one_clf=True)
        
        count += 1

        #{"class_probabiltiies": {'nudity': 0.2, 'violence': 0.2, 'gambling': 0.2, 'drugs': 0.2, 'negative': 0.2}}
        json_to_return = {'class_probabiltiies': {cl: proba for cl, proba in zip(different_classes, probas)}}
        print(json_to_return)
        return jsonify(json_to_return)
        #return jsonify(list(probas[0]))
    else:
        return '' #jsonify({'No results'})

    #return json.dumps(input_json)

if __name__ == '__main__':
    count = 0

    different_classes = ['violence', 'gambling', 'drugs', 'nudity', 'negative']

    with open('api_keys.json') as f:
        api_keys = json.load(f)

    clarifai_app = ClarifaiApp(api_key=api_keys['CLARIFAI_API_KEY'])

    # get the general model
    model = clarifai_app.models.get("general-v1.3")

    model_obj = load_model('latest_model_obj.pkl')
    clf = model_obj['clf']
    vocab = model_obj['vocabulary']
    
    print('Vocab length: {}'.format(len(vocab)))
    
    model_obj_multi_clf = load_model('latest_model_obj_5_clfs.pkl')
    
    clf_violence = model_obj_multi_clf['violence']
    clf_gambling = model_obj_multi_clf['gambling']
    clf_drugs = model_obj_multi_clf['drugs']
    clf_nudity = model_obj_multi_clf['nudity']
    clf_negative = model_obj_multi_clf['negative']
    vocab = model_obj['vocabulary']

    print('Vocab length: {}'.format(len(vocab)))

    vectorizer = CountVectorizer(vocabulary=vocab)

    app.run()