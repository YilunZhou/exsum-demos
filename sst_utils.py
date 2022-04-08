
import pickle, random
from copy import deepcopy as copy

from exsum import Parameter, ParameterRange, BehaviorRange, Data, Model

def parametrized_b_func_range(lo_lo, lo_hi, lo_val, hi_lo, hi_hi, hi_val):
	def b_func(u, params):
		lo, hi = params
		return BehaviorRange([(lo, hi)])
	b_params = [Parameter('saliency lower range', ParameterRange(lo_lo, lo_hi), lo_val),
				Parameter('saliency upper range', ParameterRange(hi_lo, hi_hi), hi_val)]
	return b_func, b_params

def load_data(split_idx, split):
	assert split in ['first', 'second']
	exp_data = pickle.load(open('sst_explanation.pkl', 'rb'))
	sentence_feus, exp_measure = exp_data['sentence_feus'], exp_data['exp_measure']
	random.seed(0)
	random.shuffle(sentence_feus)
	if split == 'first':
		data = Data(sentence_feus[:split_idx], exp_measure, normalize=True)
	else:
		data = Data(sentence_feus[split_idx:], exp_measure, normalize=True)
	return data

def load_model(rule_union, split_idx=300, split='first'):
	data = load_data(split_idx, split)
	model = Model(rule_union, data)
	return model

def precedence_seq(seq):
	'''
	convert a list of rules into a precedence composition
	e.g. if seq is [3, 5, 2, 1], then return (((3, '>', 5), '>', 2), '>', 1)
	'''
	assert len(seq) > 0, '"seq" cannot be empty'
	seq = copy(seq)
	if len(seq) == 1:
		return seq[0]
	else:
		last = seq.pop()
		return (precedence_seq(seq), '>', last)
