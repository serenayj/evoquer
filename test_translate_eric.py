import sys
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
from src.utils import io_utils, eval_utils
from pipeline_utils import *
from datetime import date 
today = str(date.today()) 
"""
Import built modules  
"""
#from VPMT_NOVSE import VPMT 
from VPMT_eric import VPMT 

from vpmt_config import * 
	
global dataset 

settings = "test_translate_eric_"+str(today) 

if __name__ == "__main__":

	#dataset = sys.argv[1] 
	dataset = "charades"
	#sys.path.append("/Users/yanjungao/Desktop/VPMT/")
	
	pretrained_lgi_path = "charades_eric_32f_200_vpmt.pkl" 
	#pretrained_lgi_path = None 
	if dataset == "charades": 
		pip_config = {
		"img_dim": 1024, 
		"img_embed_size": 1000,
		"use_abs": False, 
		"word_dim": 300,
		"text_embed_size":1000,
		"no_imgnorm": True,
		"sos_id": 2,
		"eos_id": 3,
		"decoder_max_len": 8,
		"batch_size": 64,
		}
		from src.dataset.charades import * 
		config_path = "ymls/config_char.yml"
		full_config= io_utils.load_yaml(config_path)
		config = io_utils.load_yaml(config_path)["test_loader"]
		test_D = CharadesDataset(config)
		m_config = model_args(full_config, pip_config) # this has to be full model 
		model = VPMT(m_config,"char")
		if pretrained_lgi_path != None: 
			model.load_state_dict(torch.load(pretrained_lgi_path))
			print("Succesfully load pre-trained parameters")
	else:
		pip_config = {
		"img_dim": 500, 
		"img_embed_size": 500,
		"use_abs": False, 
		"word_dim": 300,
		"text_embed_size":500,
		"no_imgnorm": True,
		"sos_id": 2,
		"eos_id": 3,
		"decoder_max_len": 8,
		"batch_size": 64,
		}
		from src.dataset.anet import * 
		config_path = "ymls/config_anet.yml"
		full_config= io_utils.load_yaml(config_path)
		config = io_utils.load_yaml(config_path)["train_loader"]
		train_D = ActivityNetCaptionsDataset(config)
		config = io_utils.load_yaml(config_path)["test_loader"]
		test_D = ActivityNetCaptionsDataset(config)

	output = {}
	for item in test_D:
		_id = item[0]['vids'] + "_" + item[0]['qids']
		vis_data = [item]
		net_inps, gts = model.LGI_model.prepare_batch_w_pipline(vis_data, False)
		lgi_out = model.LGI_model(net_inps) 
		durations = torch.tensor(gts['duration']).unsqueeze(-1).cuda()
		v_feats = extract_frames(lgi_out['grounding_loc'], net_inps['video_feats'], durations)
		src_out, init_hidden, vid_out = model.forward_vse_emb(v_feats, net_inps['query_labels'], net_inps['query_lengths'])
		gts_queries = translate_gts(gts['qids'], model.train_ids, model.test_ids, 10)
		seq, pred_lengths = model.decoder.beam_decoding(net_inps['query_labels'].long(), init_hidden, src_out, vid_out, 10)
		out_str = " ".join(model.itow[i.item()] for i in seq[0])
		output[_id] = out_str

	with open(settings + ".json","w") as outf:
		json.dump(output, outf)



