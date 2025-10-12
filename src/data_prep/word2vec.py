import gensim.downloader as api
from scipy.spatial.distance import cosine
import itertools
import numpy as np

def load_model(size='small'):
    if size == 'small':
        model_name = 'glove-twitter-25'

    if size == 'large':
        model_name = 'word2vec-google-news-300'
    print(f'down/Loading {model_name}...')
    try:
        model = api.load(model_name)
        print(f'down/Load Complete.')

    except ValueError:
        print("Could not load model. Check your internet connection or try another model.")

    return model

def measure_similarity(classes, model = None):
    if model == None:
        model = load_model('small')

    vectors = {}
    for cl in classes:
        processed_cl = cl.replace('-', '_')
        words = processed_cl.split('_')
        class_vector = np.zeros(model.vector_size)
        found_words = 0

        for word in words:
            try:
                class_vector += model[word.lower()]
                found_words += 1
            except KeyError:
                print(f"Word '{word}' from class '{cl}' not found in model.")
        if found_words > 0:
            vectors[cl] = class_vector
    
    pairs = itertools.combinations(vectors.keys(), 2)

    score = 0
    for class1, class2 in pairs:
        vec1 = vectors[class1]
        vec2 = vectors[class2]

        similarity = 1 - cosine(vec1, vec2)
        score += similarity
        # print(f"Similarity between '{class1}' and '{class2}': {similarity:.4f}")

    return score/len(classes)

if __name__ == "__main__":
    classes = ['duck','cat','bird']
    score = measure_similarity(classes)

    
