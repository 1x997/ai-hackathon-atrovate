import argparse
import glob
from pprint import pprint
import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

concept_names_for_crime = ['knife', 'crime', 'weapon', 'force', 'fight', 'sword',
                               'offense', 'danger', 'war', 'battle', 'rebellion', 'crisis',
                               'knife blade', 'stab', 'cut', 'law enforcement', 'military', 'angry', 'aggression']

def get_data(json_fp):
    with open(json_fp, 'r') as fp:
        result = json.load(fp)

    #pprint(result)
    #print()

    return get_sentence_for_result_and_concepts(result)

def get_sentence_for_result_and_concepts(result):
    all_concepts = result['outputs'][0]['data']['concepts']
    print(all_concepts)
    print('Number of concepts found: {}'.format(len(all_concepts)))

    for concept in all_concepts:
        print(concept)

    concept_names_found = [x['name'] for x in all_concepts]
    crime_concepts_found = [x for x in concept_names_found if x.lower() in concept_names_for_crime]

    print('Crime concepts found:')
    for c_c_f in crime_concepts_found:
        print(c_c_f)

    names_and_scores = [{x['name'], x['value']} for x in all_concepts]

    sentence_for_result = ' '.join(concept_names_found)

    return sentence_for_result
    #return [x['name']]

def get_sentences_from_all_image_results_in_folder(folder_path):
    all_json_paths = glob.glob(folder_path)

    all_sentences = []

    for json_path in all_json_paths:
        print('Reading json path: {}'.format(json_path))
        result = get_data(json_path)
        all_sentences.append(result)

    print('\n\nAll Results:::::::')
    print(all_sentences)

    return all_sentences

def get_X_y_from_sentences(specific_sentences, vocabulary, y_class):

    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X = vectorizer.fit_transform(specific_sentences)
    print(vectorizer.get_feature_names())

    indices_for_crime_features = [i for i, feat_name in enumerate(vectorizer.get_feature_names()) if feat_name in concept_names_for_crime]

    X_full = X.toarray()
    print(X_full)
    print(X_full.shape)

    return X_full, np.full((X_full.shape[0], 1), y_class)

    #import pdb;pdb.set_trace()

    '''sum_words = X_full.sum(axis=0)
    #words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = [(word, sum_words[idx]) for idx, word in enumerate(vectorizer.get_feature_names())]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    
    X_crime = X_full[:, indices_for_crime_features]

    print(X_crime.shape)

    return X_full, X_crime, np.ones((X_crime.shape[0], 1))'''

def get_X_y_from_folder(specific_sentences, vocabulary, y_class=0):
    #all_sentences = get_sentences_from_all_image_results_in_folder(folder_path)
    X, y = get_X_y_from_sentences(specific_sentences, vocabulary, y_class)

    return X, y

if __name__ == "__main__":
    """
        python clarifai_run_on_folder.py --input_image_folder images/gambling --class_type gambling
    """

    '''parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_results_folder", default="", help="")
    args = parser.parse_args()

    print(args)'''

    #get_data('sample_clarifai_result_1.json')

    #X_violence_full, X_violence, y_violence = get_sentences_from_all_image_results_in_folder('results/violence_image_results/*')

    violence_glob_path = 'results/violence_image_results/*'
    gambling_glob_path = 'results/gambling_image_results/*'
    drugs_glob_path = 'results/drugs_image_results/*'
    # nudity_glob_path = 'results/nudity_image_results/*'
    # negative_glob_path = 'results/negative_image_results/*'
    
    all_sentences_violence = get_sentences_from_all_image_results_in_folder(violence_glob_path)
    all_sentences_gambling = get_sentences_from_all_image_results_in_folder(gambling_glob_path)
    all_sentences_drugs = get_sentences_from_all_image_results_in_folder(drugs_glob_path)

    all_sentences = []
    all_sentences.extend(all_sentences_violence)
    all_sentences.extend(all_sentences_gambling)
    all_sentences.extend(all_sentences_drugs)

    vocabulary = list(set([word for sent in all_sentences for word in sent.split(' ')]))


    X_violence, y_violence = get_X_y_from_folder(all_sentences_violence, vocabulary, y_class=0)
    X_gambling, y_gambling = get_X_y_from_folder(all_sentences_gambling, vocabulary, y_class=1)
    X_drugs, y_drugs = get_X_y_from_folder(all_sentences_drugs, vocabulary, y_class=2)

    #X_nudity, y_nudity = get_X_y_from_folder(nudity_glob_path)
    #X_negative, y_negative = get_X_y_from_folder(negative_glob_path, y_class=1)

    #import pdb;pdb.set_trace()

    #print(result['outputs'][0]['data'])
    target_names = ['Violence', 'Gambling', 'Drugs']

    print(X_violence.shape, X_gambling.shape, X_drugs.shape)

    X_all = np.concatenate([X_violence, X_gambling, X_drugs])
    y_all = np.concatenate([y_violence, y_gambling, y_drugs])

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    print('Train X shape: {}. Test X shape: {}'.format(X_train.shape, X_test.shape))

    #clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf = LogisticRegression()
    # GaussianNB would be good.

    clf.fit(X_train, y_train)

    print('Train score: {}'.format(clf.score(X_train, y_train)))
    print('Test score: {}'.format(clf.score(X_test, y_test)))

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=target_names))

    import pdb;pdb.set_trace()