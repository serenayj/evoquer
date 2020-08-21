import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


class model_args(object):
	"""docstring for model_args"""
	def __init__(self, lgi_arg, config):
		super(model_args, self).__init__()
		self.lgi_arg = lgi_arg 
		self.init(config)
		#self.config = config 

	def init(self,config):
		"""
		Parameters initialization for models except LGI  
		"""
		self.img_dim = config["img_dim"]
		self.img_embed_size = config["img_embed_size"]
		self.use_abs = config["use_abs"]
		self.word_dim = config["word_dim"]
		self.text_embed_size = config["text_embed_size"]
		self.no_imgnorm = config["no_imgnorm"]
		self.sos_id = config["sos_id"]
		self.eos_id = config["eos_id"]
		self.max_len= config["decoder_max_len"]
		self.tie_weights = True 
		self.cuda = False 
		self.batch_size = 10
		self.epoch = 1 
		self.use_attn= True  
		self.bidirectional = True   
		self.vocab_size = 6101

		



		