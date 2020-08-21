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
from vse_video_enc import EncoderVideoC3D, EncoderText
import seq2seq 

from seq2seq.models.DecoderRNN import DecoderRNN
from pipeline_utils import *

"""
Loss Func
"""
import VSE 
from VSE.model import ContrastiveLoss
from seq2seq.loss import NLLLoss 

class VPMT(nn.Module):
	"""docstring for Pipeline"""
	def __init__(self, arg):
		self.vocab_size = 6101
		super(VPMT, self).__init__()
		self.arg = arg
		self.LGI_arg = arg.lgi_arg # LGI model uses its own parameters
		#self.LGI_model = LGI(arg) 
		self.init_LGI()
		self.weight_loss = True # if using weighted loss 
		self.VSE_vdo_enc = EncoderVideoC3D(arg.img_dim, arg.img_embed_size,\
								use_abs=arg.use_abs,\
								no_imgnorm=arg.no_imgnorm,\
								use_bi=self.arg.bidirectional)
		self.VSE_txt_enc = EncoderText(self.vocab_size, arg.word_dim,\
							   arg.text_embed_size,\
							   use_abs=arg.use_abs,\
							   use_bi=self.arg.bidirectional)
		if self.arg.tie_weights: # Use same embedding layer for LGI and VSE 
			self.VSE_txt_enc.embedding = self.LGI_model.query_enc.embedding 


		if self.arg.cuda:
			self.LGI_model.cuda() 
			self.VSE_vdo_enc.cuda()
			self.VSE_txt_enc.cuda()

		#self.loss_fn = ComplexLoss() 
		self.get_parameters()
		self.vseloss = ContrastiveLoss()

	def get_method(self,method_type):
		if method_type.startswith("tgn"):
			M = bn.get_temporal_grounding_network(None, method_type, True)
		else:
			raise NotImplementedError("Not supported model type ({})".format(method_type))
		return M

	def train_mode(self):
		self.LGI_model.train_mode()
		self.VSE_vdo_enc.train() 
		self.VSE_txt_enc.train()
		self.LGI_model.reset_status() # initialize status

	def eval_mode(self):
		self.LGI_model.eval_mode()
		self.VSE_vdo_enc.eval() 
		self.VSE_txt_enc.eval()


	def init_LGI(self):
		M = self.get_method("tgn_lgi") # import module 
		self.LGI_model = M.LGI(self.LGI_arg) 

	def get_parameters(self):
		self.LGI_params = list(self.LGI_model.get_parameters())
		self.VSE_enc_params = list(self.VSE_vdo_enc.parameters()) + list(self.VSE_txt_enc.parameters())
		self.model_params = self.LGI_params + self.VSE_enc_params 

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

	def compute_loss_vse(self, v_emb, q_emb):
		"""
		Compute loss from LGI   
		"""
		b = v_emb.shape[0] # batch size 
		self.vloss = self.vseloss(v_emb, q_emb)  

	def combine_loss(self):
		if self.weight_loss:
			self.total_loss = 10*self.lgi_loss['total_loss']+self.vloss
		else:
			self.total_loss = self.lgi_loss['total_loss'] + self.vloss 
	
	def get_lr(self):
		for param_group in self.optimizer.param_groups:
			return param_group["lr"]

	def update_lr(self):
		cur_lr = self.optimizer.param_groups[0]['lr']
		self.optimizer.param_groups[0]['lr']= cur_lr * 0.1
		print("========= UPDATE LR RATE AT {} =========".format(cur_lr/5))

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

	def forward_vse_emb(self, images, captions, lengths, volatile=False):
		"""Compute the video and query embeddings
		"""
		# Set mini-batch dataset
		images = torch.tensor(images)
		captions = torch.tensor(captions)
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()

		# Forward
		#print("images shape ", images.shape)
		img_emb, img_out = self.VSE_vdo_enc(images)
		cap_emb, cap_out = self.VSE_txt_enc(captions, lengths)
		return img_emb, cap_emb, img_out, cap_out 


	def forward(self, net_inps, gts):
		""" input: batch net_inps
			pipeline forward:1. LGI
							 2. GET VIDEO FEATS FROM PREDICTED LOC 
							 3. VSE 
							 4. Decoder 
		"""
		""" Step 1 & 2 """
		self.LGI_model.reset_status() 
		lgi_out = self.LGI_model(net_inps) 
		self.lgi_out = lgi_out
		self.LGI_model.compute_status(lgi_out, gts) 
		self.compute_loss_lgi(lgi_out, gts)
		v_feats = extract_frames(lgi_out['grounding_loc'], net_inps['video_feats'])
		self.v_feats = v_feats
		""" Step 3 """
		v_emb, cap_emb, img_out, cap_out = self.forward_vse_emb(v_feats, net_inps['query_labels'], net_inps['query_lengths'])
		self.compute_loss_vse(v_emb, cap_emb)
		self.combine_loss()
		return self.total_loss

	def save_model(self, path):
		torch.save(self.state_dict(), path)

	def print_info_but_lgi(self, mode, epoch, _iter, logger=None):
		txt = "[== VMPT ALL ==][{}] {} epoch {} iter".format(mode, epoch, _iter)
		txt += ", TOTAL LOSS = {:.4f}, VSE LOSS = {:.4f}, LGI Loss = {:.4f}".format(self.total_loss, self.vloss, self.total_loss.item() - self.vloss.item())
		if logger:
			logger.info(txt)
		else:
			print(txt)

	def translate(self, itow, gts, prefix="", write=False):
		pred_words = self.decode_out[-1]['sequence']
		preds = torch.stack(pred_words,dim=0) 
		preds = preds.transpose(1,0)
		outs = {} 
		B = len(gts['query_labels'])
		uniscs = []
		biscs = [] 
		for _ in range(B):
			vid = gts["vids"][_] 
			if vid not in outs:
				outs[vid] = {} 
			qid = gts["qids"][_] 
			outs[vid][qid] = []  
			gt_words =[itow[item.item()] for item in gts['query_labels'][_] if item.item() in itow and item.item()!=0]
			out_ids = [item.item() for item in preds[_]] 
			pred_words = [itow[item] for item in out_ids if item !=0] 
			unigram = nltk.translate.bleu_score.sentence_bleu([gt_words], pred_words, weights=[1])
			bigram = nltk.translate.bleu_score.sentence_bleu([gt_words], pred_words, weights=(0,1,0,0)) 
			uniscs.append(unigram)
			biscs.append(bigram) 
			outs[vid][qid] = [" ".join(gt_words), " ".join(pred_words), unigram, bigram] 
		self.outs = outs
		import numpy as np 
		print("Average Unigram BLEU Score: {}, Bigram BLEU Score: {} ".format(np.mean(uniscs), np.mean(biscs)))
		if write:
			with open(prefix+"translate_out.csv", "w") as outf:
				wr = csv.writer(outf)
				for k, v in bot.model.outs.items():
					for kk in v:
						item = [k, kk, v[kk]]
						wr.writerow(item)

"""
Uni-test purpose
"""
from vpmt_config import * 
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
	import sys
	sys.path.append("/Users/yanjungao/Desktop/VPMT/")
	from src.utils import io_utils, eval_utils
	config_path="/Users/yanjungao/Desktop/LGI4temporalgrounding-master/pretrained_models/charades_LGI/config.yml"
	full_config= io_utils.load_yaml(config_path)
	config = io_utils.load_yaml(config_path)["train_loader"]
	from src.dataset.charades import * 
	D = CharadesDataset(config)
	m_config = model_args(full_config, pip_config) # this has to be full model 
	vpmt_pip = VPMT(m_config) 
	vis_data = D.get_samples(int(4))
	net_inps, gts = vpmt_pip.LGI_model.prepare_batch_w_pipline(vis_data, False)
	lgi_out = vpmt_pip.LGI_model(net_inps) 
	"""
	vpmt_pip.LGI_model.compute_status(lgi_out, gts) 
	v_feats = extract_frames(lgi_out['grounding_loc'], net_inps['video_feats'])
	vpmt_pip.v_feats = v_feats
	v_emb, cap_emb, img_out, cap_out  = vpmt_pip.forward_vse_emb(v_feats, net_inps['query_labels'], net_inps['query_lengths'])
	mix_emb, cap_out = vpmt_pip.forward_encode(net_inps['description_labels'], 
	"""











