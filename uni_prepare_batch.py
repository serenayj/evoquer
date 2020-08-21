def preapre_batch_w_pipline(batch):
	input_keys = ['vids', 'qids', 'timestamps', 'duration', 'description_length', 'description_labels', 'query_lengths', 'query_labels', 'query_masks', 'grounding_start_pos', 'grounding_end_pos', 'grounding_att_masks', 'nfeats', 'video_feats', 'video_masks']
	net_inps_keys = ['description_length', 'description_labels', 'query_lengths', 'query_labels', 'query_masks', 'grounding_att_masks', 'video_feats', 'video_masks']
	gt_list = ["vids", "qids", "timestamps", "duration",
			   "grounding_start_pos", "grounding_end_pos",
			   "grounding_att_masks", "nfeats"]
	both_list = ["grounding_att_masks"]
	tensor_inps_keys = ['query_lengths', 'query_labels', 'query_masks', 'grounding_att_masks', 'video_feats', 'video_masks']

	net_inps, gts = {}, {}
	for k in net_inps_keys:
		net_inps[k] = []  
		for items in batch:
			for subitem in items:
				val = subitem[k] 
				net_inps[k].append(val)


input_keys = ['vids', 'qids', 'timestamps', 'duration', 'description_length', 'description_labels', 'query_lengths', 'query_labels', 'query_masks', 'grounding_start_pos', 'grounding_end_pos', 'grounding_att_masks', 'nfeats', 'video_feats', 'video_masks']
net_inps_keys = ['description_length', 'description_labels', 'query_lengths', 'query_labels', 'query_masks', 'grounding_att_masks', 'video_feats', 'video_masks']
gt_list = ["vids", "qids", "timestamps", "duration",
		   "grounding_start_pos", "grounding_end_pos",
		   "grounding_att_masks", "nfeats"]
both_list = ["grounding_att_masks"]
tensor_inps_keys = ['description_labels', 'query_labels', 'query_masks', 'grounding_att_masks', 'video_feats', 'video_masks']

net_inps, gts = {}, {}
for k in net_inps_keys:
	net_inps[k] = []  
	for items in batch:
		for subitem in items:
			val = subitem[k] 
			net_inps[k].append(val)
			
for k in gt_list:
	gts[k] = []  
	for items in batch:
		for subitem in items:
			val = subitem[k] 
			gts[k].append(val)

for k in tensor_inps_keys:
	print(k)
	val = net_inps[k]
	if torch.is_tensor(val[0]):
		net_inps[k] = torch.stack(val).squeeze(1)
		gts[k] = torch.stack(val).squeeze(1)
		#net_inps[k] = torch.stack(val).squeeze(1).to(self.device)
	else:
		net_inps[k] = torch.tensor(val)
		gts[k] =torch.tensor(val)
		#net_inps[k] = torch.tensor(val).to(self.device)

if use_tag_loss:
	gts["tag_att_masks"] = gts["grounding_att_masks"]



