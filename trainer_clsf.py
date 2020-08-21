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
import json
from src.experiment import common_functions as cmf
from src.utils import timer

from datetime import date 

"""
Import built modules  
"""
from VPCLS import VPCLS
from label_loader import * 
#from classifiers_sigmoid import AcClassifier, ObjClassifier 
today = str(date.today()) 

class SupervisedTrainer(object):
	"""docstring for SupervisedTrainer"""
	def __init__(self, full_cfg, lgi_config, config, train_db, test_db):
		super(SupervisedTrainer, self).__init__()
		self.lgi_config = lgi_config
		self.config = config 
		self.train_db = train_db 
		self.test_db = test_db
		self.label_data = LabelMaker2("") 
		
		""" Model building """
		self.model =  VPCLS(full_cfg, self.label_data.verb_vocab, self.label_data.noun_vocab)
		self.model.create_optimizer() 
		self.lr_update = 30 # update every n epoch 
		self.batch_size = full_cfg.batch_size
		self.epoch = full_cfg.epoch 
		
		self.lr_adj = full_cfg.lr_adj  
		self.prefix = "verb_obj_recog_lr_ones_"+str(self.lr_update)
		self.init_save_dir() 
		self.save_loss = 1000
		self.best_score = 0 
		self.translate_every = full_cfg.translate_every
	
	def init_save_dir(self):
		#today = str(date.today()) 
		self.model.LGI_model.config["misc"]["result_dir"] = self.model.LGI_model.config["misc"]["result_dir"]+self.prefix+today 
	
	def train(self):
		""" Prepare work  """
		cmf.create_save_dirs(self.lgi_config['misc']) # LGI only 
		self.train_db_lst = len(self.train_db)
		it_logger = cmf.create_logger(self.lgi_config, "ITER", "train.log")
		eval_logger = cmf.create_logger(self.lgi_config, "EPOCH", "scores.log")


		""" Run training network """
		eval_every = self.lgi_config["evaluation"].get("every_eval", 1) # epoch
		eval_after= self.lgi_config["evaluation"].get("after_eval", 0) # epoch
		print_every = self.lgi_config["misc"].get("print_every", 1) # iteration
		num_step = self.lgi_config["optimize"].get("num_step", 30) # epoch
		apply_cl_after = self.lgi_config["model"].get("curriculum_learning_at", -1)

		# We evaluate initialized model
		#cmf.test(config, L["test"], net, 0, eval_logger, mode="Valid")
		ii = 1
		self.model.train_mode() # set network as train mode
		tm = timer.Timer() # tm: timer
		
		n_iters = int(self.train_db_lst / self.batch_size) 
		print("=====> # of iteration per one epoch: {}".format(n_iters))

		for epoch in range(0, self.epoch):
			if epoch != 0 and epoch % self.lr_update ==0 and self.lr_adj:
				self.model.update_lr()

			permutation = np.random.permutation(self.train_db_lst) # D, length of D 
			permutation = list(map(int,permutation))
			self.permutation = permutation
			
			self.model.train_mode()

			print("Shuffling batch with {} iterations ".format(n_iters))

			for _iter in range(n_iters):
				batched = [] 
				for _id in permutation[_iter*self.batch_size: (_iter+1)*self.batch_size]:
					item = self.train_db[_id]
					batched.append(item)

				# Forward and update the network
				data_load_duration = tm.get_duration()
				tm.reset()
				net_inps, gts = self.model.LGI_model.prepare_batch_w_pipline(batched,self.model.arg.cuda)

				if self.model.sigmoid:
					loss = self.model(net_inps, gts, self.label_data.train_verb_ones, self.label_data.train_obj_id)

				self.model.update()

				run_duration = tm.get_duration()

				# Compute status for current batch: loss, evaluation scores, etc
				self.model.LGI_model.compute_status(self.model.lgi_out, gts)

				# print learning status
				if (print_every > 0) and (ii % print_every == 0):
					self.model.LGI_model.print_status()
					lr = self.model.get_lr()
					txt = "fetching for {:.3f}s, optimizing for {:.3f}s, lr = {:.5f}"
					it_logger.info(txt.format(data_load_duration, run_duration, lr))

				tm.reset(); ii = ii + 1
				# iteration done

			# validate current model
			if (epoch > eval_after) and (epoch % eval_every == 0):
			#if (epoch == 0 ) and (epoch % eval_every == 0):
				# print training losses
				self.model.print_info_but_lgi("Train", epoch, _iter, )
				self.model.LGI_model.save_results("epoch{:03d}".format(epoch), mode="Train")
				self.model.LGI_model.print_counters_info(eval_logger, epoch, mode="Train")

				self.validate_LGI(eval_logger, config, epoch)

				self.model.train_mode() # set network as train mode
				self.model.LGI_model.reset_status() # initialize status

				# Save models if best scores 
				if self.model.LGI_model.best_score > self.best_score:
					print("Saving Model with Best Scores: ", self.model.LGI_model.best_score)
					#self.save_loss = self.model.total_loss.item()
					self.model.save_model(self.prefix+"_vpcls.pkl")
					self.best_score = self.model.LGI_model.best_score

				
	def validate_LGI(self, eval_logger, config, epoch):
		self.model.eval_mode()
		self.test_db_lst = len(self.test_db)
		n_iters = int(self.test_db_lst / self.batch_size)
		batches = [] 
		permutation = list(range(self.test_db_lst))
		#for _iter in range(n_iters):
		batched = [] 
		for _iter in range(n_iters):
			batched = [] 
			for _id in permutation[_iter*self.batch_size: (_iter+1)*self.batch_size]:
				item = self.test_db[_id]
				batched.append(item)
			batches.append(batched)
		cmf.test(self.lgi_config, batches, self.model.LGI_model, epoch, eval_logger, mode="Valid")



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
	"batch_size": 64,
	}
	import sys
	#sys.path.append("/Users/yanjungao/Desktop/VPMT/")
	from src.utils import io_utils, eval_utils
	#config_path="/Users/yanjungao/Desktop/LGI4temporalgrounding-master/pretrained_models/charades_LGI/config.yml"
	config_path = "ymls/config.yml"
	full_config= io_utils.load_yaml(config_path)
	config = io_utils.load_yaml(config_path)["train_loader"]
	from src.dataset.charades import * 
	train_D = CharadesDataset(config)
	config = io_utils.load_yaml(config_path)["test_loader"]
	test_D = CharadesDataset(config)
	m_config = model_args(full_config, pip_config) # this has to be full model 
	bot = SupervisedTrainer(m_config,m_config.lgi_arg, config, train_D, test_D)
	bot.train()
