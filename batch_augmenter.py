import torch
import numpy as np 
from pipeline_utils import slice_range
from collections import Counter 

def label_batch_index(class_targets):
	""" Return dictionary: key is the label name, value is a list of index that labels in the batch"""
	batch_info = {} 
	for _, l in enumerate(class_targets):
		if l in batch_info:
			batch_info[l].append(_)
		else:
			batch_info[l] = [_]
	return batch_info 

def extract_pred_frames(pred_loc,vfeats):
	""" Select frames by predicted boundaries """
	select_vfeats = [] 
	idx = pred_loc.clone()
	idx_frames = idx * 128 
	idx_ranges, signals = slice_range(idx_frames)
	lengths = [] 
	for _id, _ in enumerate(idx_ranges):
		#lengths.append(_.numel())
		if _.numel() < 8 :
			out_feats = torch.zeros((8,1024)) 
			if max(_) > 128:
				new_ = [i for i in _ if i < 128]
				lengths.append(len(new_))
				v_feats = torch.index_select(vfeats[_id], 0, torch.tensor(new_).long())
			else:
				lengths.append(_.numel())
				v_feats = torch.index_select(vfeats[_id], 0, _.long().detach())
			#out_feats[_.long()] = v_feats  
			out_feats[:v_feats.shape[0]] = v_feats 
			select_vfeats.append(out_feats)
			#lengths.append(_.numel())
		#elif len(v)
		else:
			#print("selected id range", _)
			#print("v feats shape", vfeats.shape)
			if max(_) >= 128:
				new_ = [i for i in _ if i < 128]
				v_feats = torch.index_select(vfeats[_id], 0, torch.tensor(new_).long())
				lengths.append(len(new_))
			else:
				lengths.append(_.numel())
				v_feats = torch.index_select(vfeats[_id], 0, _.long().detach()) 
			select_vfeats.append(v_feats)
	return select_vfeats, lengths 


def video_length(vfeats):
	""" Measure nonzero length of video features """
	cnt = 0 
	for r in vfeats:
		cnt +=1 
		if r.sum() == 0 :
			return cnt 
	return cnt 

def augmenter_per_sample(class_vfeats, lengths, class_target_num):
	""" Making enough sample for a specific class, input is a list of vfeats """
	extra_data = []
	#length = video_length(vfeats) 
	#real_v_feats = vfeats[:length]
	for _ in range(class_target_num):
		if _ < len(lengths):
			length = lengths[_]
			vfeats = class_vfeats[_]
			
		else:
			_id = np.random.choice(len(lengths),1)
			length = lengths[_id.item()] 
			vfeats = class_vfeats[_id.item()]
		
		#print("length : ", length)
		if length >0:
			sample_idx = torch.randint(0,length,(8,)).sort()[0]
			sample_vf = vfeats[sample_idx]
		else:
			sample_idx = torch.zeros((8,)).long()
			sample_vf = torch.randn((8,1024))  
		#print("length ", length)
		#print("sample idx", sample_idx)
		#print("v feats shape ", vfeats.shape)
		#sample_vf = vfeats[sample_idx]
		extra_data.append(sample_vf)
	return extra_data 

def augmenter_per_batch(pred_locs, vfeats, class_targets):
	"""Input:predicted boundaries, video features, ground truth class labels  
	   Output: augmented batch with balanced label distribution   
	"""
	gts = [] 
	output_feats = [] 
	c = Counter(class_targets.tolist()) 
	#print(c)
	target_num = max(c.values())
	#print(target_num) 
	batch_label_info = label_batch_index(class_targets.tolist())
	for k,v in batch_label_info.items():
		pred_loc = pred_locs[v].clone()
		pred_vfeats, lengths = extract_pred_frames(pred_loc, vfeats)
		balanced_data = augmenter_per_sample(pred_vfeats, lengths, target_num)
		gts.extend([k] * target_num)
		output_feats.append(torch.stack(balanced_data))
	out_feats = torch.cat(output_feats)
	return out_feats, torch.tensor(gts) 




