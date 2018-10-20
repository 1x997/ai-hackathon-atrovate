from pprint import pprint
import json

with open('sample_clarifai_result_1.json', 'r') as fp:
    result = json.load(fp)

pprint(result)
print()

def print_crime_concepts(result):
    all_concepts = result['outputs'][0]['data']['concepts']
    print(all_concepts)
    print('Number of concepts found: {}'.format(len(all_concepts)))

    for concept in all_concepts:
        print(concept)

    concept_names_found = [x['name'] for x in all_concepts]
    concept_names_for_crime = ['knife', 'crime', 'weapon', 'force', 'fight', 'sword',
                               'offense', 'danger']

    crime_concepts_found = [x for x in concept_names_found if x.lower() in concept_names_for_crime]

    print('Crime concepts found:')
    for c_c_f in crime_concepts_found:
        print(c_c_f)

#print(result['outputs'][0]['data'])