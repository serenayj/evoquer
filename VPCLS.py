import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import nltk 

"""
Import built modules  
"""
import src
from src.dataset import anet, charades 
from src.model.LGI import LGI
from src.model import building_networks as bn
from src.utils import utils, io_utils

#from VSE.model import EncoderImage, EncoderText
from vse_video_enc import EncoderVideoC3D 
from classifiers_sigmoid import AcClassifier, ObjClassifier 

from pipeline_utils import *
from batch_augmenter import * 

from collections import Counter 


def convert_ids(idx, length):
	tmp = list([0]*length) 
	for _id in idx:
		tmp[_id] = 1 
	return tmp

class VPCLS(nn.Module):
	"""docstring for Pipeline"""
	def __init__(self, arg, action_vocab, obj_vocab):
		self.vocab_size = 6101
		super(VPCLS, self).__init__()
		self.arg = arg
		self.LGI_arg = arg.lgi_arg # LGI model uses its own parameters
		#self.LGI_model = LGI(arg) 
		self.init_LGI()
		self.weight_loss = False # if using weighted loss 
		self.data_augmented = False # If using data augmenter 
		self.ac_null = 169 
		self.noun_null = 381
		self.sigmoid = True 
		self.VSE_vdo_enc = EncoderVideoC3D(arg.img_dim, arg.img_embed_size,\
								use_abs=arg.use_abs,\
								no_imgnorm=arg.no_imgnorm,\
								use_bi=self.arg.bidirectional)
		self.ac_vcb_size = len(action_vocab) + 1 # 169 
		self.obj_vcb_size = len(obj_vocab)  # 383  
		""" Action verb classification """
		if self.arg.bidirectional:
			self.action_clfs = AcClassifier(2*arg.img_embed_size, self.ac_vcb_size, arg.img_embed_size)
			""" Object verb classification """
			self.obj_clfs = ObjClassifier(2*arg.img_embed_size, self.obj_vcb_size, arg.img_embed_size)
		else:
			self.action_clfs = AcClassifier(arg.img_embed_size, self.ac_vcb_size, arg.img_embed_size)
			""" Object verb classification """
			self.obj_clfs = ObjClassifier(arg.img_embed_size, self.obj_vcb_size, arg.img_embed_size)

		if self.arg.cuda:
			self.LGI_model.cuda() 
			self.VSE_vdo_enc.cuda()
			self.action_clfs.cuda()
			self.obj_clfs.cuda() 

		#self.loss_fn = ComplexLoss() 
		self.get_parameters()

	def get_method(self,method_type):
		if method_type.startswith("tgn"):
			M = bn.get_temporal_grounding_network(None, method_type, True)
		else:
			raise NotImplementedError("Not supported model type ({})".format(method_type))
		return M

	def train_mode(self):
		self.LGI_model.train_mode()
		self.VSE_vdo_enc.train() 
		self.action_clfs.train()
		self.obj_clfs.train() 
		self.LGI_model.reset_status() # initialize status

	def eval_mode(self):
		self.LGI_model.eval_mode()
		self.VSE_vdo_enc.eval() 
		self.action_clfs.eval()
		self.obj_clfs.eval() 

	def init_LGI(self):
		M = self.get_method("tgn_lgi") # import module 
		self.LGI_model = M.LGI(self.LGI_arg, self.vocab_size) 

	def get_parameters(self):
		self.LGI_params = list(self.LGI_model.get_parameters())
		self.VSE_enc_params = list(self.VSE_vdo_enc.parameters()) 
		self.clfs_params = list(self.action_clfs.parameters()) + list(self.obj_clfs.parameters())
		self.model_params = self.LGI_params + self.VSE_enc_params + self.clfs_params

	def compute_loss_cls(self, output, target_variable, mode="action"):
		"""
		Compute loss from CrossEntropy   
		"""
		if self.sigmoid:
			criterion = nn.BCELoss()  
			loss = criterion(output, target_variable.float())
		else:
			loss = F.cross_entropy(output, target_variable)
		return loss 

	def compute_loss_lgi(self, net_outs, gts):
		"""
		Compute loss from LGI   
		"""
		if torch.is_tensor(gts["grounding_end_pos"]) == False:
			gts["grounding_end_pos"] = torch.tensor(gts["grounding_end_pos"])
			gts["grounding_start_pos"] = torch.tensor(gts["grounding_start_pos"])
		if self.arg.cuda:
			gts["grounding_end_pos"].cuda()
			gts["grounding_start_pos"].cuda() 
		self.lgi_loss = self.LGI_model.criterion(net_outs, gts)  

	def combine_loss(self, weights=1):
		if self.weight_loss:
			self.total_loss = weights*self.lgi_loss['total_loss']+ self.ac_loss + self.nn_loss 
		else:
			self.total_loss = self.lgi_loss['total_loss']+ self.ac_loss + self.nn_loss 
	
	def get_lr(self):
		for param_group in self.optimizer.param_groups:
			return param_group["lr"]

	def update_lr(self):
		cur_lr = self.optimizer.param_groups[0]['lr']
		self.optimizer.param_groups[0]['lr']= cur_lr * 0.1
		print("========= UPDATE LR RATE AT {} =========".format(cur_lr/self.arg.lr_step))

	def create_optim_seperate(self):
		# raw_optimizer = optim.Adam([
		# 		{'params': self.net.cnn.parameters(), 'lr': self.cfg.finetune_lr}, 
		# 		{'params': self.net.bilstm.parameters(), 'lr': self.cfg.finetune_lr},
		# 		{'params': self.net.locnet.parameters(), 'lr': self.cfg.finetune_lr},
		# 		{'params': self.net.scalenet.parameters(), 'lr': self.cfg.finetune_lr}
		# 	], lr=self.cfg.lr)
		# optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
		# scheduler = optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=3, gamma=0.8)
		# optimizer.set_scheduler(scheduler)
		raise NotImplementedError

	def create_optimizer(self):
		lr =  self.LGI_arg["optimize"]["init_lr"]
		opt_type = self.LGI_arg["optimize"]["optimizer_type"]
		if opt_type == "SGD":
			self.optimizer = torch.optim.SGD(
					self.model_params, lr=lr,
					momentum=self.LGI_arg["optimize"]["momentum"],
					weight_decay=self.LGI_arg["optimize"]["weight_decay"])
		elif opt_type == "Adam":
			betas = self.LGI_arg["optimize"].get("betas", (0.9,0.999))
			weight_decay = self.LGI_arg["optimize"].get("weight_decay", 0.0)
			self.optimizer = torch.optim.Adam(
				self.model_params, lr=lr, betas=betas,
				weight_decay=weight_decay)
		elif opt_type == "Adadelta":
			self.optimizer = torch.optim.Adadelta(self.model_params, lr=lr)
		elif opt_type == "RMSprop":
			self.optimizer = torch.optim.RMSprop(self.model_params, lr=lr)
		else:
			raise NotImplementedError(
				"Not supported optimizer [{}]".format(opt_type))

		# setting scheduler
		self.scheduler = None
		scheduler_type = self.LGI_arg["optimize"].get("scheduler_type", "")
		decay_factor = self.LGI_arg["optimize"]["decay_factor"]
		decay_step = self.LGI_arg["optimize"]["decay_step"]
		if scheduler_type == "step":
			self.scheduler = torch.optim.lr_scheduler.StepLR(
					self.optimizer, decay_step, decay_factor)
		elif scheduler_type == "multistep":
			milestones = self.LGI_arg["optimize"]["milestones"]
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
					self.optimizer, milestones, decay_factor)
		elif scheduler_type == "exponential":
			self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
					self.optimizer, decay_factor)
		elif scheduler_type == "lambda":
			lambda1 = lambda it: it // decay_step
			lambda2 = lambda it: decay_factor ** it
			self.scheduler = torch.optim.lr_scheduler.LambdaLR(
					self.optimizer, [lambda1, lambda2])
		elif scheduler_type == "warmup":
			raise NotImplementedError()

	def update(self):
		""" Update the network
		Args:
			loss: loss to train the network; dict()
		"""

		#self.it = self.it + 1
		# initialize optimizer
		if self.optimizer == None:
			self.create_optimizer()
			self.optimizer.zero_grad() # set gradients as zero before update

		self.total_loss.backward()
		if self.scheduler is not None: self.scheduler.step()
		self.optimizer.step()
		self.optimizer.zero_grad()

	def forward_vse_emb(self, images, volatile=False):
		"""Compute the video and query embeddings
		"""
		# Set mini-batch dataset
		images = torch.tensor(images)
		if torch.cuda.is_available():
			images = images.cuda()

		# Forward
		img_emb, img_out = self.VSE_vdo_enc(images)
		return img_emb, img_out

	def forward_clsf(self, gts, action_gt, obj_gt, lgi_out, net_inps):
		ac_gts = torch.tensor([action_gt[i] if i in action_gt else self.ac_null for i in gts["qids"]])
		if not self.sigmoid:
			nn_gts = torch.tensor([obj_gt[i] if i in obj_gt else self.noun_null for i in gts["qids"] ])
		else:
			#nn_gts = torch.tensor(convert_ids([obj_gt[i]]) if i in obj_gt else convert_ids(self.noun_null) for i in gts["qids"]) 
			nn_gts = [] 
			for i in gts["qids"]:
				if i in obj_gt:
					nn_gts.append(convert_ids(obj_gt[i], self.obj_vcb_size)) 
				else:
					nn_gts.append(convert_ids(self.noun_null, self.obj_vcb_size))
			nn_gts = torch.tensor(nn_gts)


		if not self.data_augmented:
			v_feats = extract_frames(lgi_out['grounding_loc'], net_inps['video_feats'])
			self.v_feats = v_feats
			v_emb, img_out = self.forward_vse_emb(v_feats)
		else: 
			""" Only need one set of features, either action, or nouns  """
			ac_out_feats, ac_gts = augmenter_per_batch(lgi_out['grounding_loc'], vfeats, ac_gts) 
			nn_out_feats, nn_gts = augmenter_per_batch(lgi_out['grounding_loc'], vfeats, nn_gts) 
			v_emb, img_out = self.forward_vse_emb(ac_out_feats) 

		self.ac_gts = ac_gts
		self.nn_gts = nn_gts
		if torch.cuda.is_available():
			ac_gts = ac_gts.cuda()
			nn_gts = nn_gts.cuda()

		ac_preds = self.action_clfs(v_emb)
		nn_preds = self.obj_clfs(v_emb) 

		self.ac_preds = ac_preds 
		self.nn_preds = nn_preds
		self.ac_acc = self.label_accuracy(ac_preds, ac_gts)
		self.nn_acc = self.label_accuracy(nn_preds, nn_gts, mode="object") 

		self.ac_loss = self.compute_loss_cls(ac_preds, ac_gts)
		self.nn_loss = self.compute_loss_cls(nn_preds, nn_gts, mode="object")

	def forward(self, net_inps, gts, action_gt, obj_gt):
		""" input: batch net_inps
			pipeline forward:1. LGI
							 2. GET VIDEO FEATS FROM PREDICTED LOC 
							 3. VSE 
							 4. OBJ, VERB classification 
		"""
		""" Step 1 & 2 """
		self.LGI_model.reset_status() 
		lgi_out = self.LGI_model(net_inps) 
		self.lgi_out = lgi_out
		self.LGI_model.compute_status(lgi_out, gts) 
		self.compute_loss_lgi(lgi_out, gts)
		
		""" Step 3 """
		#self.forward_clsf(v_emb, gts, action_gt, obj_gt) 
		self.forward_clsf(gts, action_gt, obj_gt, lgi_out, net_inps)
		self.combine_loss() 
		self.update()

	def label_accuracy(self, preds, target_variable, mode="action"):
		""" accuracy of predicting the correct labels """
		b = preds.shape[0]
		if self.sigmoid: 
			pred_indices = torch.round(preds) # if action classification and sigmoid is using, take rounding as prediction 
			hits = torch.eq(pred_indices, target_variable.float()).sum()
		else:
			pred_indices = torch.max(preds, dim=-1)[1]
			hits = torch.eq(pred_indices, target_variable.long()).sum()
		return float(hits/b) 

	def save_model(self, path):
		torch.save(self.state_dict(), path)

	def print_info_but_lgi(self, mode, epoch, _iter, logger=None):
		txt = "[== VMPT ALL ==][{}] {} epoch {} iter".format(mode, epoch, _iter)
		txt += ", TOTAL LOSS = {:.4f}, VERB LOSS = {:.4f}, OBJ LOSS = {:.4f}".format(self.total_loss, self.ac_loss, self.nn_loss)
		if logger:
			logger.info(txt)
		else:
			print(txt)

"""
Uni-test purpose
"""
from vpmt_config import * 
from label_loader import * 
if __name__ == "__main__":
	pip_config = {
	"img_dim": 1024, 
	"img_embed_size": 1000,
	"use_abs": False, 
	"word_dim": 300,
	"text_embed_size":1000,
	"no_imgnorm": True,
	"sos_id": 2,
	"eos_id": 3,
	"decoder_max_len": 10,
	}
	label_data = LabelMaker2("/Users/yanjungao/Desktop/VPMT/") 
	import sys
	sys.path.append("/Users/yanjungao/Desktop/VPMT/")
	from src.utils import io_utils, eval_utils
	config_path="ymls/config.yml"
	full_config= io_utils.load_yaml(config_path)
	config = io_utils.load_yaml(config_path)["train_loader"]
	from src.dataset.charades import * 
	D = CharadesDataset(config)
	m_config = model_args(full_config, pip_config) # this has to be full model 
	vpmt_pip = VPCLS(m_config, label_data.verb_vocab, label_data.noun_vocab) 

	vis_data = D.get_samples(int(4))
	net_inps, gts = vpmt_pip.LGI_model.prepare_batch_w_pipline(vis_data, False)
	lgi_out = vpmt_pip.LGI_model(net_inps) 
	vpmt_pip.LGI_model.compute_status(lgi_out, gts) 
	v_feats = extract_frames(lgi_out['grounding_loc'], net_inps['video_feats'])
	vpmt_pip.v_feats = v_feats
	v_embs, v_out = vpmt_pip.forward_vse_emb(v_feats)
	vpmt_pip.forward_clsf(gts, label_data.train_verb_ones, label_data.train_obj_id, lgi_out, net_inps)











