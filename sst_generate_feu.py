
import pickle, random
from collections import Counter

import numpy as np
from tqdm import tqdm

from exsum import SentenceGroupedFEU, Measure
import sst_features

def compute_features(tokens, word_counter, num_words):
    sentiments = [sst_features.get_word_sentiment(t) for t in tokens]
    poss = sst_features.parse_pos(tokens)
    ners = sst_features.parse_ner(tokens)
    deps = sst_features.parse_dep(tokens)
    freqs = [word_counter[t] / num_words for t in tokens]
    features = list(zip(sentiments, poss, ners, deps, freqs))
    return features

def construct_sst_feus(fn):
    explanation_data = pickle.load(open(fn, 'rb'))
    all_words = [w for e in explanation_data for w in e['sentence']]
    word_counter = Counter(all_words)
    num_words = len(all_words)
    sentences = []
    all_explanations = []
    all_weights = []
    for e in tqdm(explanation_data):
        tokens = e['sentence']
        features = compute_features(tokens, word_counter, num_words)
        explanations = e['saliency']
        true_label = e['label']
        pred_label = e['prediction']
        sentences.append(SentenceGroupedFEU(tokens, features, explanations, 
            true_label, pred_label))
        all_explanations.extend(explanations)
        all_weights.extend([1 / len(explanations)] * len(explanations))
    exp_measure = Measure(np.array(all_explanations), np.array(all_weights), 
        zero_discrete=True)
    return sentences, exp_measure

if __name__ == '__main__':
    in_fn = 'sst_explanation_raw.pkl'
    out_fn = 'sst_explanation.pkl'
    sentence_feus, exp_measure = construct_sst_feus(in_fn)
    pickle.dump({'sentence_feus': sentence_feus, 'exp_measure': exp_measure}, 
        open(out_fn, 'wb'))
