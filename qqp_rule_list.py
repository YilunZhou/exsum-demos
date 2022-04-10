
from collections import Counter

from exsum import Parameter, ParameterRange, BehaviorRange, Rule

from qqp_utils import parametrized_b_func_range

def get_split_idx(u):
	all_qids = [f[0] for f in u.context.features]
	split_idx = len(all_qids) * 2 - sum(all_qids)
	return split_idx

def rule_match_neg_pred(idx):
	def ab_func(u, a_params, b_params):
		'''
		requirements:
		1. POS needs to be one of 'PROPN', 'NOUN', 'ADJ', 'VERB', 'PRON'
		2. the parent question cannot contain this word as duplicate
		3. the word has exactly one case-insensitive match in the other question
		4. only applies to negative sentences?
		'''
		lo, hi = b_params
		w = u.word.lower()
		assert lo <= 0 and hi >= 0
		qid, pos, ner, dep, freq = u.feature
		if pos not in ['PROPN', 'NOUN', 'ADJ', 'VERB', 'PRON'] or u.prediction >= 0.5:
			return False, BehaviorRange.simple_interval(0, 0)
		all_words = [t.lower() for t in u.context.words]
		split_idx = get_split_idx(u)
		q1_words = all_words[:split_idx]
		q2_words = all_words[split_idx:]
		if Counter(q1_words)[w] != 1 or Counter(q2_words)[w] != 1:
			return False, BehaviorRange.simple_interval(0, 0)
		if qid == 1:
			v = u.context.explanations[split_idx + q2_words.index(w)]
		else:
			v = u.context.explanations[q1_words.index(w)]
		return True, BehaviorRange.simple_interval(v + lo, v + hi)
	a_params = []
	b_params = [Parameter('lower margin', ParameterRange(-1, 0), -0.07),
				Parameter('upper margin', ParameterRange(0, 1), 0.07)]
	return Rule([idx, 'matching words have matching saliency for negative preds', \
		ab_func, a_params, b_params])

def rule_match_pos_pred(idx):
	def ab_func(u, a_params, b_params):
		'''
		requirements:
		1-3: same as above
		4. only applies to positive sentences
		'''
		lo, hi = b_params
		w = u.word.lower()
		assert lo <= 0 and hi >= 0
		qid, pos, ner, dep, freq = u.feature
		if pos not in ['PROPN', 'NOUN', 'ADJ', 'VERB', 'PRON'] or u.prediction < 0.5:
			return False, BehaviorRange.simple_interval(0, 0)
		all_words = [t.lower() for t in u.context.words]
		split_idx = get_split_idx(u)
		q1_words = all_words[:split_idx]
		q2_words = all_words[split_idx:]
		if Counter(q1_words)[w] != 1 or Counter(q2_words)[w] != 1:
			return False, BehaviorRange.simple_interval(0, 0)
		if qid == 1:
			v = u.context.explanations[split_idx + q2_words.index(w)]
		else:
			v = u.context.explanations[q1_words.index(w)]
		return True, BehaviorRange.simple_interval(v + lo, v + hi)
	a_params = []
	b_params = [Parameter('lower margin', ParameterRange(-1, 0), -0.18),
				Parameter('upper margin', ParameterRange(0, 1), 0.18)]
	return Rule([idx, 'matching words have matching saliency for positive preds', \
		ab_func, a_params, b_params])

def rule_non_match_neg_pred(idx):
	def ab_func(u, a_params, b_params):
		'''
		requirements:
		1. POS needs to be one of 'PROPN', 'NOUN', 'ADJ', 'VERB', 'PRON'
		2. the word has exactly one case-insensitive match in the other question
		3. only applies to negative sentences?
		'''
		lo, hi = b_params
		w = u.word.lower()
		assert lo <= 0 and hi >= 0
		qid, pos, ner, dep, freq = u.feature
		if pos not in ['PROPN', 'NOUN', 'VERB', 'ADJ', 'PRON'] or u.prediction >= 0.5:
			return False, BehaviorRange.simple_interval(0, 0)
		all_words = [t.lower() for t in u.context.words]
		split_idx = get_split_idx(u)
		q1_words = all_words[:split_idx]
		q2_words = all_words[split_idx:]
		if qid == 1:
			if Counter(q2_words)[w] >= 1:
				return False, BehaviorRange.simple_interval(0, 0)
			else:
				return True, BehaviorRange.simple_interval(lo, hi)
		else:  # qid == 2
			if Counter(q1_words)[w] >= 1:
				return False, BehaviorRange.simple_interval(0, 0)
			else:
				return True, BehaviorRange.simple_interval(lo, hi)
	a_params = []
	b_params = [Parameter('lower range', ParameterRange(-1, 0), -0.35),
				Parameter('upper range', ParameterRange(0, 1), 0.01)]
	return Rule([idx, 'nonmatching words for negative preds', \
		ab_func, a_params, b_params])

def rule_non_match_pos_pred(idx):
	def ab_func(u, a_params, b_params):
		'''
		requirements:
		1. POS needs to be one of 'PROPN', 'NOUN', 'ADJ', 'VERB', 'PRON'
		2. the word has exactly one case-insensitive match in the other question
		3. only applies to negative sentences?
		'''
		lo, hi = b_params
		w = u.word.lower()
		assert lo <= 0 and hi >= 0
		qid, pos, ner, dep, freq = u.feature
		if pos not in ['PROPN', 'NOUN'] or u.prediction < 0.5:
			return False, BehaviorRange.simple_interval(0, 0)
		all_words = [t.lower() for t in u.context.words]
		split_idx = get_split_idx(u)
		q1_words = all_words[:split_idx]
		q2_words = all_words[split_idx:]
		if qid == 1:
			if Counter(q2_words)[w] >= 1:
				return False, BehaviorRange.simple_interval(0, 0)
			else:
				return True, BehaviorRange.simple_interval(lo, hi)
		else:  # qid == 2
			if Counter(q1_words)[w] >= 1:
				return False, BehaviorRange.simple_interval(0, 0)
			else:
				return True, BehaviorRange.simple_interval(lo, hi)
	a_params = []
	b_params = [Parameter('lower range', ParameterRange(-1, 0), -0.31),
				Parameter('upper range', ParameterRange(0, 1), 0.01)]
	return Rule([idx, 'nonmatching words for positive preds', \
		ab_func, a_params, b_params])

def rule_question_mark_neg_pred(idx):
	'''This rule is not used due to being very not sharp'''
	def a_func(u, a_params):
		if u.word != '?' or u.prediction >= 0.5:
			return False
		split_idx = get_split_idx(u)
		if u.idx not in [split_idx - 1, u.L - 1]:
			return False
		return True
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.04, -1, 1, 0.03)
	return Rule([idx, 'terminal question mark for negative preds', \
		a_func, a_params, b_func, b_params])

def rule_question_mark_pos_pred(idx):
	def a_func(u, a_params):
		if u.word != '?' or u.prediction < 0.5:
			return False
		split_idx = get_split_idx(u)
		if u.idx not in [split_idx - 1, u.L - 1]:
			return False
		return True
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.07, -1, 1, 0.06)
	return Rule([idx, 'terminal question mark for positive preds', \
		a_func, a_params, b_func, b_params])

def rule_stop_words_neg_pred(idx):
	def ab_func(u, a_params, b_params):
		w = u.word.lower()
		qid, pos, ner, dep, freq = u.feature
		if u.prediction >= 0.5:
			return False, BehaviorRange.simple_interval(-0.06, 0.025)
		if u.word.lower() == 'do':
			return True, BehaviorRange.simple_interval(-0.09, 0.02)
		elif pos == 'ADP':
			return True, BehaviorRange.simple_interval(-0.06, 0.035)
		elif pos in ['AUX', 'DET', 'SCONJ', 'CCONJ', 'PUNCT']:
			return True, BehaviorRange.simple_interval(-0.06, 0.025)
		else:
			return False, BehaviorRange.simple_interval(-0.06, 0.025)
	return Rule([idx, 'stop words for negative preds', ab_func, [], []])

def rule_stop_words_pos_pred(idx):
	def ab_func(u, a_params, b_params):
		w = u.word.lower()
		qid, pos, ner, dep, freq = u.feature
		if u.prediction < 0.5:
			return False, BehaviorRange.simple_interval(-0.09, 0.08)
		if u.word.lower() == 'do':
			return True, BehaviorRange.simple_interval(-0.1, 0.06)
		elif pos == 'ADP':
			return True, BehaviorRange.simple_interval(-0.09, 0.19)
		elif pos in ['AUX', 'DET', 'SCONJ', 'CCONJ', 'PUNCT']:
			return True, BehaviorRange.simple_interval(-0.09, 0.14)
		else:
			return False, BehaviorRange.simple_interval(-0.09, 0.08)
	return Rule([idx, 'stop words for positive preds', ab_func, [], []])

def rule_neg_neg_pred(idx):
	def a_func(u, a_params):
		qid, pos, ner, dep, freq = u.feature
		return dep.startswith('neg') and u.prediction < 0.5
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.21, -1, 1, 0.01)
	return Rule([idx, 'negation words for negative preds', 
		a_func, a_params, b_func, b_params])

def rule_neg_pos_pred(idx):
	def a_func(u, a_params):
		qid, pos, ner, dep, freq = u.feature
		return dep.startswith('neg') and u.prediction >= 0.5
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.1, -1, 1, 0.24)
	return Rule([idx, 'negation words for positive preds', 
		a_func, a_params, b_func, b_params])

def rule_everything_else_neg_pred(idx):
	def a_func(u, a_params):
		qid, pos, ner, dep, freq = u.feature
		return u.prediction < 0.5
	def b_func(u, b_params):
		qid, pos, ner, dep, freq = u.feature
		if pos == 'NOUN':
			return BehaviorRange.simple_interval(-0.09, 0.06)
		elif pos == 'VERB':
			return BehaviorRange.simple_interval(-0.05, 0.05)
		elif pos == 'ADJ':
			return BehaviorRange.simple_interval(-0.09, 0.05)
		else:
			return BehaviorRange.simple_interval(-0.12, 0.04)
	return Rule([idx, 'everything else for negative preds', 
		a_func, [], b_func, []])

def rule_everything_else_pos_pred(idx):
	def a_func(u, a_params):
		return u.prediction >= 0.5
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.13, -1, 1, 0.25)
	return Rule([idx, 'everything else for positive preds', 
		a_func, a_params, b_func, b_params])
