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
import os 

num_frames = 24  # global: number of frames extracted for encoder 


"""
Simplfied translation functions 
"""

def init_trans(path):
	train_id_path = os.path.join(path, "train_translate.json")
	test_id_path = os.path.join(path, "test_translate.json")
	train_ids = json.load(open(train_id_path,'r'))[0]
	test_ids = json.load(open(test_id_path,'r'))[0]
	vocab_path = os.path.join(path, "vocab_translate.json")
	idx_vocab = json.load(open(vocab_path,'r'))[1]
	vocab_idx = json.load(open(vocab_path,'r'))[0]

	return train_ids, test_ids, idx_vocab, vocab_idx 

def translate_gts(gts_qids, train_ids, test_ids, max_len):
	B = len(gts_qids)
	if torch.cuda.is_available():
		outs = torch.zeros(B,max_len+2).cuda()
	else:
		outs = torch.zeros(B,max_len+2)
	if gts_qids[0] in train_ids:
		for qid in gts_qids:
			v = train_ids[qid] 
			outs[gts_qids.index(qid)][:len(v)] = torch.tensor(v) 
	else:
		for qid in gts_qids:
			v = test_ids[qid] 
			outs[gts_qids.index(qid)][:len(v)] = torch.tensor(v) 
	return outs 

"""
Import built modules  
"""

def slice_range(idx):
	""" Extend [start, end] to a list of index """
	idx = idx.long().detach()
	out_idx = [] 
	for _ in idx:
		if _[0] < _[1]:
			ttt = torch.range(_[0], _[1])
			out_idx.append(ttt)
			reverse_signal = False 
		else:
			reverse_signal = True 
			ttt = torch.tensor([_[0], _[1]])
			out_idx.append(ttt)
	return out_idx, reverse_signal   

def pad_frames(v_feats):
	tgt = torch.zeros(num_frames, 1024)
	tgt[:v_feats.shape[0],:] = v_feats 
	return tgt 

def extract_frames(pred_loc, video_feats, durations):
	"""Input: B x 2  """
	idx = pred_loc.clone() / durations
	idx_frames = idx * 128 
	idx_ranges, signals = slice_range(idx_frames)
	v_feats_range = [] 
	for _id, _ in enumerate(idx_ranges):
		#print("Extract id ranges", _)
		#print("v feats shape ", video_feats[_id].shape)
		try:
			v_feats = torch.index_select(video_feats[_id], 0, _.long().detach()) 
			out_feats = downsample_frames(v_feats)
			if out_feats.shape[0] != num_frames:
				#print(""v_feats.shape)
				break 
			v_feats_range.append(out_feats)
		except:
			if _[0] < 128 and _[-1] >= 128:
				tmp = 128 - _[0] -1
				v_feats = torch.index_select(video_feats[_id], 0, tmp.long().detach()) 
				out_feats = downsample_frames(v_feats)
			else:
				out_feats = torch.zeros((num_frames,1024))
			
			v_feats_range.append(out_feats)
		#print("out feature shape ", out_feats.shape)
	return torch.stack(v_feats_range, dim=0)

def downsample_frames(v_feats):
	length = v_feats.shape[0] 
	if length < num_frames:
		out_feats = pad_frames(v_feats)
		return out_feats 
	if length == num_frames :
		return v_feats 
	sample_idx = torch.range(0, v_feats.shape[0]-1, (v_feats.shape[0]-1) / (num_frames-1)).long() 
	out_feats = torch.index_select(v_feats, 0, sample_idx.detach())
	if len(sample_idx) < num_frames:
		out_feats = pad_frames(out_feats)
	return out_feats 
	


