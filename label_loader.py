import json 
import os 

class LabelMaker(object):
	"""docstring for LabelMaker"""
	def __init__(self, path):
		super(LabelMaker, self).__init__()
		self.path = path
		self.train_verb_id_path = os.path.join(path, "train_verb_idx.json")
		self.test_verb_id_path = os.path.join(path, "test_verb_idx.json")
		self.train_obj_id_path = os.path.join(path, "train_obj_idx.json")
		self.test_obj_id_path =  os.path.join(path, "test_obj_idx.json")
		self.verb_vocab_path = os.path.join(path, "verb_vocab.json")
		self.noun_vocab_path = os.path.join(path, "obj_vocab.json") 
		self.load_data() 
		self.load_vocab()

	def load_data(self):
		self.train_verb_id, self.train_verb_tk = json.load(open(self.train_verb_id_path, 'r'))
		self.test_verb_id, self.test_verb_tk = json.load(open(self.test_verb_id_path, 'r'))
		self.train_obj_id, self.train_obj_tk = json.load(open(self.train_obj_id_path, 'r'))
		self.test_obj_id, self.test_obj_tk = json.load(open(self.test_obj_id_path, 'r'))


	def load_vocab(self):
		self.verb_vocab, self.verb_vc_ind = json.load(open(self.verb_vocab_path, 'r')) 
		self.noun_vocab, self.noun_vc_ind = json.load(open(self.noun_vocab_path, 'r')) 




class LabelMaker2(object):
	"""docstring for LabelMaker"""
	def __init__(self, path):
		super(LabelMaker2, self).__init__()
		self.path = path
		self.train_verb_id_path = os.path.join(path, "train_verb_idx_syn.json")
		self.test_verb_id_path = os.path.join(path, "test_verb_idx_syn.json")
		self.train_obj_id_path = os.path.join(path, "train_obj_idx.json")
		self.test_obj_id_path =  os.path.join(path, "test_obj_idx.json")
		self.verb_vocab_path = os.path.join(path, "verb_vocab_syn.json")
		self.noun_vocab_path = os.path.join(path, "vocab_obj.json") 
		self.load_data() 
		self.load_vocab()

	def load_data(self):
		self.train_verb_id, self.train_verb_ones, self.train_verb_tk = json.load(open(self.train_verb_id_path, 'r'))
		self.test_verb_id, self.test_verb_ones, self.test_verb_tk = json.load(open(self.test_verb_id_path, 'r'))
		self.train_obj_id, self.train_obj_tk = json.load(open(self.train_obj_id_path, 'r'))
		self.test_obj_id, self.test_obj_tk = json.load(open(self.test_obj_id_path, 'r'))

	def load_vocab(self):
		self.verb_vocab, self.verb_vc_ind = json.load(open(self.verb_vocab_path, 'r')) 
		self.noun_vc_ind, self.noun_vocab  = json.load(open(self.noun_vocab_path, 'r'))
		
if __name__ == '__main__':
	label_loader = LabelMaker2("")
	print(label_loader.noun_vocab)
