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
from src.experiment import common_functions as cmf
from src.utils import timer
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

from pipeline_utils import *

import seq2seq 
from seq2seq.models.DecoderRNN import DecoderRNN

"""
Loss Func
"""
import VSE 
from VSE.model import ContrastiveLoss
from seq2seq.loss import NLLLoss 
from datetime import date 
today = str(date.today())

#torch.cuda.set_device(0)

class VSE_DECODE(nn.Module):
	"""docstring for Pipeline"""
	def __init__(self, arg, glove_vecs):
		#self.vocab_size = len(glove_vecs)
		self.vocab_size = arg.vocab_size
		self.glove = torch.tensor(glove_vecs)
		super(VSE_DECODE, self).__init__()
		self.arg = arg
		self.LGI_arg = arg.lgi_arg
		self.VSE_vdo_enc = EncoderVideoC3D(arg.img_dim, arg.img_embed_size,\
								use_abs=arg.use_abs,\
								no_imgnorm=arg.no_imgnorm,\
								use_bi=self.arg.bidirectional)
		self.VSE_txt_enc = EncoderText(self.vocab_size, arg.word_dim,\
							   arg.text_embed_size,\
							   use_abs=arg.use_abs,\
							   use_bi=self.arg.bidirectional)

		# Load glove and freeze weights 
		if arg.bidirectional:
			self.fusion = nn.Linear(2*2*arg.text_embed_size, 2*arg.text_embed_size) 
		else:
			# if no bidirectional 
			self.fusion = nn.Linear(arg.text_embed_size+arg.img_embed_size, arg.text_embed_size)

		self.use_attn = self.arg.use_attn

		if self.arg.use_attn:
			self.decoder = DecoderRNN(self.vocab_size, arg.max_len, arg.text_embed_size,
			arg.sos_id, arg.eos_id, bidirectional=arg.bidirectional, use_attention=True)
		else:	
			self.decoder = DecoderRNN(self.vocab_size, arg.max_len, arg.text_embed_size,
			arg.sos_id, arg.eos_id,bidirectional=arg.bidirectional)

		#self.load_glove() 

		if self.arg.cuda:
			self.VSE_vdo_enc.cuda()
			self.VSE_txt_enc.cuda()
			self.fusion.cuda()
			self.decoder.cuda() 

		#self.loss_fn = ComplexLoss() 
		self.get_parameters()
		self.nllloss = NLLLoss()
		self.vseloss = ContrastiveLoss()

	def load_glove(self):
		"""
		Load glove vector tensor to embedding weights and freeze these params  
		"""
		self.VSE_txt_enc.embed.weight.data = self.glove 
		self.VSE_txt_enc.freeze_emb_params()
		#self.decoder.embedding.weight.data = self.glove 
		#self.decoder.freeze_emb_params()

	def train_mode(self):
		self.VSE_vdo_enc.train() 
		self.VSE_txt_enc.train()
		self.fusion.train()
		self.decoder.train() 

	def eval_mode(self):
		self.VSE_vdo_enc.eval() 
		self.VSE_txt_enc.eval()
		self.fusion.eval()
		self.decoder.eval() 

	def get_parameters(self):
		self.VSE_enc_params = list(self.VSE_vdo_enc.parameters()) + list(self.VSE_txt_enc.parameters())
		self.model_params = self.VSE_enc_params  
		self.fusion_params = list(self.fusion.parameters())
		self.decoder_params = list(self.decoder.parameters()) 
		self.model_params = self.VSE_enc_params + self.fusion_params + self.decoder_params 

	def combine_loss(self):
		#self.total_loss = self.nllloss.loss + self.v_loss
		self.total_loss = self.nllloss.loss 
	def compute_loss_nll(self, decoder_outputs, target_variable):
		"""
		Compute loss from NLL  
		"""
		self.nllloss.reset() 
		for step, step_output in enumerate(decoder_outputs):
			batch_size = target_variable.size(0)
			self.nllloss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1]) 
		self.nllloss.loss = self.nllloss.acc_loss / self.nllloss.norm_term

	def compute_loss_vse(self, v_emb, q_emb):
		"""
		Compute loss from LGI   
		"""
		b = v_emb.shape[0] # batch size 
		self.v_loss = self.vseloss(v_emb, q_emb) / b  

	def get_lr(self):
		for param_group in self.optimizer.param_groups:
			return param_group["lr"]

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

		self.v_loss.backward()
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

	def forward_encode(self, descriptions, lengths, video_emb, volatile=False):
		"""Compute encoding for descriptions and video embedding 
		"""
		descriptions = Variable(descriptions, volatile=volatile)
		if torch.cuda.is_available():
			video_emb = video_emb.cuda()
			descriptions = descriptions.cuda()
		# Forward
		cap_emb, cap_out = self.VSE_txt_enc(descriptions, lengths)
		outt = torch.cat((cap_emb, video_emb), dim=-1) # B X (H_video + H_txt)
		hidden_out = self.fusion(outt) # Back to B X H
		hidden_out = torch.tanh(hidden_out)
		return hidden_out, cap_out

	def forward_decode(self,inputs, encoder_hidden, encoder_outputs):
		"""Decoder decodes from encoder output, both descriptions and videos 
		"""
		#B = encoder_hidden.shape[0]
		#input_vars =  torch.tensor([vpmt_pip.arg.sos_id]*B).unsqueeze(-1).long()

		if len(encoder_hidden.shape) == 2:
			encoder_hidden = encoder_hidden.unsqueeze(0) # 1 X B X 1000 
		decoder_outputs, decoder_hidden, ret_dict = self.decoder(None, encoder_hidden, encoder_outputs)
		return decoder_outputs, decoder_hidden, ret_dict


	def forward(self, net_inps, gts):
		""" input: batch net_inps
			pipeline forward:1. LGI
							 2. GET VIDEO FEATS FROM PREDICTED LOC 
							 3. VSE 
							 4. Decoder 
		"""
		""" Step 1 & 2 """
		v_feats = extract_frames(net_inps['grounding_loc'], net_inps['video_feats'])
		self.v_feats = v_feats
		""" Step 3 """
		v_emb, cap_emb, img_out, cap_out = self.forward_vse_emb(v_feats, net_inps['query_labels'], net_inps['query_lengths'])
		self.compute_loss_vse(v_emb, cap_emb)
		if self.arg.cuda:
			net_inps['description_labels'].cuda()
		mix_emb, cap_out = self.forward_encode(net_inps['description_labels'], net_inps['description_length'], v_emb)
		self.mix_emb = mix_emb
		sent_output = nn.utils.rnn.pad_packed_sequence(cap_out) # B X L X 2H 
		sent_out = sent_output[0].transpose(1,0)
		decode_out = self.forward_decode(None, self.mix_emb, sent_out)
		self.decode_out = decode_out
		self.compute_loss_nll(decode_out[0], gts['query_labels'])
		self.combine_loss()

		return self.total_loss 

	def save_model(self, path):
		torch.save(self.state_dict(), path)

	def print_info_but_lgi(self, mode, epoch, _iter, logger=None):
		txt = "[== VSE DECODE ==][{}] {} epoch {} iter".format(mode, epoch, _iter)
		txt += ", TOTAL LOSS = {:.4f} VSE LOSS = {:.4f} NLL LOSS = {:.4f} ".format(self.total_loss, self.v_loss, self.nllloss.loss)
		if logger:
			logger.info(txt)
		else:
			print(txt)

	def train(self, train_db, test_db, itow):
		self.batch_size = 64
		self.epoch = 120
		self.train_db = train_db
		self.test_db = test_db
		self.train_db_lst = len(train_db)
		self.best_loss = 1000 
		self.itow = itow
		""" Prepare work  """
		cmf.create_save_dirs(self.LGI_arg['misc']) # LGI only 
		train_db_lst = len(self.train_db)
		it_logger = cmf.create_logger(self.LGI_arg, "ITER", "train_vse.log")
		eval_logger = cmf.create_logger(self.LGI_arg, "EPOCH", "scores_vse.log")

		""" Run training network """
		eval_every = self.LGI_arg["evaluation"].get("every_eval", 1) # epoch
		eval_after= self.LGI_arg["evaluation"].get("after_eval", 0) # epoch
		print_every = self.LGI_arg["misc"].get("print_every", 1) # iteration
		num_step = self.LGI_arg["optimize"].get("num_step", 30) # epoch
		apply_cl_after = self.LGI_arg["model"].get("curriculum_learning_at", -1)

		# We evaluate initialized model
		#cmf.test(config, L["test"], net, 0, eval_logger, mode="Valid")
		ii = 1
		self.train_mode() # set network as train mode
		tm = timer.Timer() # tm: timer
		
		n_iters = int(self.train_db_lst / self.batch_size) 
		print("=====> # of iteration per one epoch: {}".format(n_iters))

		for epoch in range(0, self.epoch):

			permutation = np.random.permutation(self.train_db_lst) # D, length of D 
			permutation = list(map(int,permutation))
			self.permutation = permutation
			
			self.train_mode()

			print("Shuffling batch with {} iterations ".format(n_iters))

			for _iter in range(n_iters):
				batched = [] 
				for _id in permutation[_iter*self.batch_size: (_iter+1)*self.batch_size]:
					item = self.train_db[_id]
					batched.append(item)

				# Forward and update the network
				data_load_duration = tm.get_duration()
				tm.reset()
				net_inps, gts = prepare_batch_w_pipline(batched,self.arg.cuda)
				net_inps = orgainze_grounding_loc(net_inps, gts)
				loss = self.forward(net_inps, gts)
				self.update()

				run_duration = tm.get_duration()
			
				# print learning status
				if (print_every > 0) and (ii % print_every == 0):
					self.print_info_but_lgi("Train", epoch, _iter,)
					lr = self.get_lr()
					txt = "fetching for {:.3f}s, optimizing for {:.3f}s, lr = {:.5f}"
					it_logger.info(txt.format(data_load_duration, run_duration, lr))

				tm.reset(); ii = ii + 1
				# iteration done
			self.validate_translate(eval_logger,self.LGI_arg,epoch)
			# Save models if best scores 
			if loss < self.best_loss:
				self.best_loss = self.v_loss 
				print("Saving Model with Best Scores: ", self.best_loss)
				#self.save_loss = self.model.total_loss.item()
				self.save_model(today+"_vse_decode.pkl")

	def validate_translate(self, eval_logger, config, epoch):
		self.eval_mode()
		self.test_db_lst = len(self.test_db)
		n_iters = int(self.test_db_lst / self.batch_size)
		batches = [] 
		permutation = list(range(self.test_db_lst))
		for _iter in range(n_iters):
			batched = [] 
			for _id in permutation[_iter*self.batch_size: (_iter+1)*self.batch_size]:
				item = self.test_db[_id]
				batched.append(item)
			batches.append(batched)
		net_inps, gts = prepare_batch_w_pipline(batched,self.arg.cuda)
		net_inps = orgainze_grounding_loc(net_inps, gts)
		self.gts = gts 
		loss = self.forward(net_inps, gts)
		self.translate(gts, "{}".format(epoch))

	def translate(self, gts, prefix="", write=False):
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
			gt_words =[self.itow[item.item()] for item in gts['query_labels'][_] if item.item() in self.itow and item.item()!=0]
			out_ids = [item.item() for item in preds[_]] 
			pred_words = [self.itow[item] for item in out_ids if item !=0] 
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
				for k, v in outs.items():
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
	import json 
	from src.utils import io_utils, eval_utils
	config_path="ymls/config.yml"
	full_config= io_utils.load_yaml(config_path)
	config = io_utils.load_yaml(config_path)["train_loader"]
	from src.dataset.charades import * 
	D = CharadesDataset(config)
	config = io_utils.load_yaml(config_path)["test_loader"]
	test_D = CharadesDataset(config)
	m_config = model_args(full_config, pip_config) # this has to be full model
	from pipeline_utils import prepare_batch_w_pipline  
	#gl_vecs = load_glove("glove_db.json")  
	model = VSE_DECODE(m_config, None)
	model.create_optimizer()
	import json 
	#data = json.load(open("glove_db.json"))
	#data_dic = data['wtoi']

	#itow = {v:k for k,v in data_dic.items()}
	#model.train(D, test_D, itow)


