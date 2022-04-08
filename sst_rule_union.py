
from exsum import RuleUnion

import os, sys
sys.path.append(os.path.dirname(__file__))
from sst_rule_list import *
from sst_utils import load_model, precedence_seq


rules = [rule_negation(1), 
         rule_positive_adj(2), # not used, merged into Rule 4
         rule_negative_adj(3), 
         rule_positive_words(4), 
         rule_negative_words(5), 
         rule_person_name(6), 
         rule_stop_words(7), # not used, replaced with Rule 8-18
         rule_stop_words_the(8), 
         rule_stop_words_a(9), 
         rule_stop_words_an(10),
         rule_stop_words_of(11), 
         rule_stop_words_pos_and_pos(12), 
         rule_stop_words_neg_and_neg(13), 
         rule_stop_words_PRON(14),
         rule_stop_words_PART(15), 
         rule_stop_words_ADP(16), 
         rule_stop_words_PUNCT(17), 
         rule_stop_words_all(18),
         rule_zero_sentiment(19), 
         rule_mildly_positive_words(20), 
         rule_mildly_negative_words(21)
        ]

stop_words = precedence_seq([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
composition_structure = precedence_seq([1, 4, 3, 5, 6, stop_words, 19, 20, 21])
rule_union = RuleUnion(rules, composition_structure)
model = load_model(rule_union, split_idx=300)
