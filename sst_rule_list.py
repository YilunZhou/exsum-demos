
from exsum import Parameter, ParameterRange, Rule

from sst_utils import parametrized_b_func_range

def rule_negation(idx):
	def a_func(u, params):
		return (u.word.lower() in ['n\'t', 'not', 'no', 'nothing']) or \
			('neg' in u.feature[3])
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -1, -1, 1, 0.002)
	return Rule([idx, 'negation', a_func, a_params, b_func, b_params])

def rule_positive_adj(idx):
	def a_func(u, params):
		threshold, = params
		assert threshold <= 1, 'min positivity should be less than 1'
		sentiment, pos, ner, dep, freq = u.feature
		return sentiment >= threshold and pos == 'ADJ'
	a_params = [Parameter('sentiment threshold', ParameterRange(0.0, 1.0), 0.37)]
	b_func, b_params = parametrized_b_func_range(0, 1, 0.01, 0, 1, 1)
	return Rule([idx, 'highly positive adjectives have positive saliency', \
		   a_func, a_params, b_func, b_params])

def rule_negative_adj(idx):
	def a_func(u, params):
		threshold, = params
		assert threshold <= 1, 'min positivity should be less than 1'
		sentiment, pos, ner, dep, freq = u.feature
		return sentiment <= threshold and pos == 'ADJ'
	a_params = [Parameter('sentiment threshold', ParameterRange(-1, 0), -0.1)]
	b_func, b_params = parametrized_b_func_range(-1, 0, -1, -1, 0, -0.06)
	return Rule([idx, 'highly negative adjectives have negative saliency', \
		   a_func, a_params, b_func, b_params])

def rule_positive_words(idx):
	def a_func(u, params):
		threshold, = params
		assert threshold <= 1, 'min positivity should be less than 1'
		sentiment, pos, ner, dep, freq = u.feature
		return sentiment >= threshold
	a_params = [Parameter('sentiment threshold', ParameterRange(0.0, 1.0), 0.39)]
	b_func, b_params = parametrized_b_func_range(0, 1, 0.01, 0, 1, 1)
	return Rule([idx, 'highly positive words have positive saliency', \
		   a_func, a_params, b_func, b_params])

def rule_negative_words(idx):
	def a_func(u, params):
		threshold, = params
		assert threshold <= 1, 'min positivity should be less than 1'
		sentiment, pos, ner, dep, freq = u.feature
		return sentiment <= threshold
	a_params = [Parameter('sentiment threshold', ParameterRange(-1, 0), -0.34)]
	b_func, b_params = parametrized_b_func_range(-1, 0, -1, -1, 0, -0.01)
	return Rule([idx, 'highly negative words have negative saliency', \
		   a_func, a_params, b_func, b_params])

def rule_person_name(idx):
	def a_func(u, params):
		sentiment, pos, ner, dep, freq = u.feature
		return 'PERSON' in ner
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.06, -1, 1, 0.1)
	return Rule([idx, 'Person names have small saliency', \
		   a_func, a_params, b_func, b_params])

def rule_stop_words(idx):
	def a_func(u, params):
		sentiment, pos, ner, dep, freq = u.feature
		return pos in ['AUX', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'PRON', 'PART', 'PUNCT']
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.1, -1, 1, 0.13)
	return Rule([idx, 'Stop words have small saliency', \
		   a_func, a_params, b_func, b_params])

def rule_stop_words_the(idx):
	def a_func(u, params):
		return u.word.lower() == 'the'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.03, -1, 1, 0.09)
	return Rule([idx, 'Stop word "the"', a_func, a_params, b_func, b_params])

def rule_stop_words_a(idx):
	def a_func(u, params):
		return u.word.lower() == 'a'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.05, -1, 1, 0.1)
	return Rule([idx, 'Stop word "a"', a_func, a_params, b_func, b_params])

def rule_stop_words_an(idx):
	def a_func(u, params):
		return u.word.lower() == 'an'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.1, -1, 1, 0.14)
	return Rule([idx, 'Stop word "an"', a_func, a_params, b_func, b_params])

def rule_stop_words_of(idx):
	def a_func(u, params):
		return u.word.lower() == 'of'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.08, -1, 1, 0.06)
	return Rule([idx, 'Stop word "of"', a_func, a_params, b_func, b_params])

def rule_stop_words_pos_and_pos(idx):
	def a_func(u, params):
		e = u.context.explanations
		return u.word.lower() == 'and' and 0 < u.idx < u.L - 1 and e[u.idx-1] > 0 \
			and e[u.idx+1] > 0
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, 0.02, -1, 1, 0.13)
	return Rule([idx, 'Stop word "and" in positive context', 
		a_func, a_params, b_func, b_params])

def rule_stop_words_neg_and_neg(idx):
	def a_func(u, params):
		e = u.context.explanations
		return u.word.lower() == 'and' and 0 < u.idx < u.L - 1 and e[u.idx-1] < 0 \
			and e[u.idx+1] < 0
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, 0.01, -1, 1, 0.1)
	return Rule([idx, 'Stop word "and" in negative context', 
		a_func, a_params, b_func, b_params])

def rule_stop_words_PRON(idx):
	def a_func(u, params):
		_, pos, _, _, _ = u.feature
		return pos == 'PRON'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.07, -1, 1, 0.15)
	return Rule([idx, 'Stop word PRON', a_func, a_params, b_func, b_params])

def rule_stop_words_PART(idx):
	def a_func(u, params):
		_, pos, _, _, _ = u.feature
		return pos == 'PART'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.08, -1, 1, 0.09)
	return Rule([idx, 'Stop word PART', a_func, a_params, b_func, b_params])

def rule_stop_words_ADP(idx):
	def a_func(u, params):
		_, pos, _, _, _ = u.feature
		return pos == 'ADP'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.1, -1, 1, 0.11)
	return Rule([idx, 'Stop word ADP', a_func, a_params, b_func, b_params])

def rule_stop_words_PUNCT(idx):
	def a_func(u, params):
		_, pos, _, _, _ = u.feature
		return pos == 'PUNCT'
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.02, -1, 1, 0.17)
	return Rule([idx, 'Stop word PUNCT', a_func, a_params, b_func, b_params])

def rule_stop_words_all(idx):
	def a_func(u, params):
		sentiment, pos, ner, dep, freq = u.feature
		return pos in ['AUX', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'PRON', 'PART', 'PUNCT']
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.1, -1, 1, 0.15)
	return Rule([idx, 'All other stop words', \
		   a_func, a_params, b_func, b_params])

def rule_zero_sentiment(idx):
	def a_func(u, params):
		sentiment, pos, ner, dep, freq = u.feature
		return sentiment == 0
	a_params = []
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.13, -1, 1, 0.15)
	return Rule([idx, 'Zero-sentiment words have small saliency', \
		   a_func, a_params, b_func, b_params])

def rule_mildly_positive_words(idx):
	def a_func(u, params):
		threshold, = params
		assert threshold <= 1, 'min positivity should be less than 1'
		sentiment, pos, ner, dep, freq = u.feature
		return 0 < sentiment < threshold
	a_params = [Parameter('sentiment threshold', ParameterRange(0.0, 1.0), 0.39)]
	b_func, b_params = parametrized_b_func_range(-1, 1, -0.11, -1, 1, 1)
	return Rule([idx, 'mildly positive words', \
		   a_func, a_params, b_func, b_params])

def rule_mildly_negative_words(idx):
	def a_func(u, params):
		threshold, = params
		sentiment, pos, ner, dep, freq = u.feature
		return threshold < sentiment < 0
	a_params = [Parameter('sentiment threshold', ParameterRange(-1, 0), -0.34)]
	b_func, b_params = parametrized_b_func_range(-1, 1, -1, -1, 1, 0.05)
	return Rule([idx, 'mildly negative words', \
		   a_func, a_params, b_func, b_params])
