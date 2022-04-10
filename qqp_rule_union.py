
from exsum import RuleUnion

import os, sys
sys.path.append(os.path.dirname(__file__))
from qqp_rule_list import *
from qqp_utils import load_model, precedence_seq


rules = [rule_match_neg_pred(1), 
         rule_match_pos_pred(2), 
         rule_non_match_neg_pred(3), 
         rule_non_match_neg_pred(4), # not used, merged into Rule 12
         rule_question_mark_neg_pred(5), 
         rule_question_mark_pos_pred(6), 
         rule_stop_words_neg_pred(7), 
         rule_stop_words_pos_pred(8), 
         rule_neg_neg_pred(9), 
         rule_neg_pos_pred(10), 
         rule_everything_else_neg_pred(11), 
         rule_everything_else_pos_pred(12)
        ]

composition_structure = precedence_seq([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12])
rule_union = RuleUnion(rules, composition_structure)
model = load_model(rule_union, split_idx=500)
