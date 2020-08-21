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

""" Pre-defined Loss modules """
import seq2seq
from seq2seq.loss import NLLLoss

import src
from  src.model.building_blocks import TAGLoss, TGRegressionCriterion 

import VSE 
from VSE.model import ContrastiveLoss


class ComplexLoss(object):
	"""
	Supervised Trainer with three loss:
		- temporal grounding loss: LGI 
		- Visual embedding loss: similarity function from VSE 
		- NLL Loss: from translation  
	"""
	def __init__(self):
		super(ComplexLoss, self).__init__()
		self.vse_loss = ContrastiveLoss()
		self.nllloss = NLLLoss() 

	def prepare_backward(self):
		loss_sum = self.nllloss.loss + self.vse_loss.loss 

