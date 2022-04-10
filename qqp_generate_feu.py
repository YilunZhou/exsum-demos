
import pickle, random
from collections import Counter

import numpy as np
from tqdm import tqdm

from exsum import SentenceGroupedFEU, Measure
import qqp_features

def compute_features(tokens, word_counter, num_words, q_idx):
    assert q_idx in [1, 2], 'q_idx is 1 or 2 and ' + \
        'specifies whether the tokens are for the 1st sentence or the 2nd sentence'
    q_idxs = [q_idx] * len(tokens)
    poss = qqp_features.parse_pos(tokens)
    ners = qqp_features.parse_ner(tokens)
    deps = qqp_features.parse_dep(tokens)
    freqs = [word_counter[t] / num_words for t in tokens]
    features = list(zip(q_idxs, poss, ners, deps, freqs))
    return features

def construct_qqp_feus(fn):
    explanation_data = pickle.load(open(fn, 'rb'))
    all_words = [w for e in explanation_data for w in e['q1_tokens'] + e['q2_tokens']]
    word_counter = Counter(all_words)
    num_words = len(all_words)
    sentences = []
    all_explanations = []
    all_weights = []
    for e in tqdm(explanation_data):
        q1_tokens = e['q1_tokens']
        q2_tokens = e['q2_tokens']
        q1_features = compute_features(q1_tokens, word_counter, num_words, 1)
        q2_features = compute_features(q2_tokens, word_counter, num_words, 2)
        explanations = e['saliency']
        assert len(q1_tokens) + len(q2_tokens) == len(explanations), \
            'something seriously wrong happens...'
        true_label = e['label']
        pred_label = e['prediction']
        sentences.append(SentenceGroupedFEU(
            q1_tokens + q2_tokens, q1_features + q2_features, 
            explanations, true_label, pred_label)
        )
        all_explanations.extend(explanations)
        all_weights.extend([1 / len(explanations)] * len(explanations))
    exp_measure = Measure(np.array(all_explanations), np.array(all_weights), 
        zero_discrete=True)
    return sentences, exp_measure

if __name__ == '__main__':
    in_fn = 'qqp_explanation_raw.pkl'
    out_fn = 'qqp_explanation.pkl'
    sentence_feus, exp_measure = construct_qqp_feus(in_fn)
    pickle.dump({'sentence_feus': sentence_feus, 'exp_measure': exp_measure}, 
        open(out_fn, 'wb'))
