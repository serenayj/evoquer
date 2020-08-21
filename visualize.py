import os
import random
import numpy as np
from tqdm import tqdm
getattr(tqdm, '_instances', {}).clear()
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt
#from IPython.display import Video, HTML

# for visualization
import cv2
from moviepy.editor import *

from src.utils import io_utils, eval_utils

np.set_printoptions(precision=3, suppress=True)

#%matplotlib inline
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)

from PIL import Image, ImageDraw, ImageFont

FAST = True

def expand_mask(mask, margin=2, height=12):
	w = mask.shape[1]
	out = [np.zeros((1,w,3), dtype=np.int) for i in range(margin)]
	for i in range(height):
		out.append(mask)
	for i in range(margin):
		out.append(np.zeros((1,w,3), dtype=np.int))
	return np.concatenate(out, axis=0)

def text_phantom(text, width=480):
	# Availability is platform dependent
	font = 'DejaVuSans-Bold'

	# Create font
	pil_font = ImageFont.truetype(font + ".ttf", size=16,
								  encoding="unic")
	text_width, text_height = pil_font.getsize(text)

	# create a blank canvas with extra space between lines
	canvas = Image.new('RGB', [width,20], (255, 255, 255))

	# draw the text onto the canvas
	draw = ImageDraw.Draw(canvas)
	white = "#000000"
	draw.text((0,0), text, font=pil_font, fill=white)

	# (text, background): (black, while) -> (white, black)
	return 255 - np.asarray(canvas)

def sampling_idx(preds, policy="random"):
	idx = random.randint(0, len(preds["gts"])-1)
	if policy == "random":
		return idx
	elif policy == "success":
		pred = preds["predictions"][idx][0]
		gt = preds["gts"][idx]
		while eval_utils.compute_tiou(pred, gt) < 0.8 or preds["gts"][idx][0] < 15:
			idx = random.randint(0, len(preds["gts"])-1)
			pred = preds["predictions"][idx][0]
			gt = preds["gts"][idx]
	elif policy == "failure":
		pred = preds["predictions"][idx][0]
		gt = preds["gts"][idx]
		while eval_utils.compute_tiou(pred, gt) > 0.2:
			idx = random.randint(0, len(preds["gts"])-1)
			pred = preds["predictions"][idx][0]
			gt = preds["gts"][idx]
	return idx

def make_bar(gt, pred, vlen, wbar):
	# draw bar for GT and prediction
	gt_idx = np.round(np.asarray(gt) / vlen * wbar).astype(np.int)
	pred_idx = np.round(np.asarray(pred) / vlen * wbar).astype(np.int)
	gt_mask, pred_mask = np.zeros((1,wbar,3)), np.zeros((1,wbar,3))
	gt_mask[0, gt_idx[0]:gt_idx[1], 0] = 255 # Red color
	pred_mask[0, pred_idx[0]:pred_idx[1], 2] = 255 # blue color

	# expand masks for better visualization and concatenate them
	bar = np.concatenate([expand_mask(gt_mask, margin=4), expand_mask(pred_mask)], axis=0)
	return bar

def make_result_video(preds, D, dt, vid_dir, policy="random", visualize=True):
	# sampling index and fetching relevant information
	#policy = "success" # among ["random" | "success" | "failure"]
	idx = sampling_idx(preds, policy)

	vlen = preds["durations"][idx]
	qid = preds["qids"][idx]
	pred = preds["predictions"][idx][0]
	gt = preds["gts"][idx]
	vid = preds["vids"][idx]
	query = " ".join(D.anns[qid]["tokens"])
	assert vid == D.anns[qid]["video_id"], "{} != {}".format(vid, D.anns[qid]["video_id"])
	assert vlen == D.anns[qid]["duration"], "{} != {}".format(vlen, D.anns[qid]["duration"])

	# concatenate two videos where one for GT (red) and another for prediction (blue)
	vw, mg, nw = 480, 20, 50 # video_width, margin, number of words at each line
	if dt == "anet":
		vname = vid[2:] + ".mp4"
	elif dt == "charades":
		vname = vid + ".mp4"
	else:
		raise NotImplementedError()
	vid_path = vid_dir + vname
	print(f"video path: {vid_path}")
	vid_GT = concatenate_videoclips([
		VideoFileClip(vid_path).subclip(0, gt[0]).margin(mg),
		VideoFileClip(vid_path).subclip(gt[0], min(gt[1],vlen)).margin(mg, color=(255,0,0)), # red
		VideoFileClip(vid_path).subclip(min(gt[1],vlen), vlen).margin(mg),
		])
	vid_pred = concatenate_videoclips([
		VideoFileClip(vid_path).subclip(0, pred[0]).margin(mg),
		VideoFileClip(vid_path).subclip(pred[0], min(pred[1],vlen)).margin(mg, color=(0,0,255)), # blue
		VideoFileClip(vid_path).subclip(min(pred[1],vlen), vlen).margin(mg),
		])
	cc = clips_array([[vid_GT, vid_pred]]).resize(width=vw)
	if FAST:
		if dt == "charades":
			factor = np.round(vlen / 20)
		else:
			factor = np.round(vlen / 30)
		print(f"speedup factor: {factor}")
		cc = cc.speedx(factor=factor)

	print(f"duration  : {vlen}")
	print(f"vid       : {vid}")
	print(f"Q         : {query}")
	print(f"prediction: {pred}")
	print(f"gt.       : {gt}")
	#cc.ipython_display(width=vw, maxduration=300)
	#cc.ipython_display(maxduration=300)

	# draw query in text image
	query = "Q: " + query
	nlines = np.int(np.ceil(len(query) / nw))
	tline = []
	for nl in range(nlines):
		if nl == nlines-1:
			cur_text = text_phantom(query[nl*nw:], vw)
		else:
			cur_text = text_phantom(query[nl*nw:nl*nw+nw], vw)
		tline.append(cur_text)
	q_text = np.concatenate(tline, axis=0)

	# draw bar for GT and prediction
	gt_idx = np.round(np.asarray(gt) / vlen * vw).astype(np.int)
	pred_idx = np.round(np.asarray(pred) / vlen * vw).astype(np.int)
	gt_mask, pred_mask = np.zeros((1,vw,3)), np.zeros((1,vw,3))
	gt_mask[0, gt_idx[0]:gt_idx[1], 0] = 255 # Red color
	pred_mask[0, pred_idx[0]:pred_idx[1], 2] = 255 # blue color
	# expand masks for better visualization and concatenate them
	bar = np.concatenate([expand_mask(gt_mask, margin=4), expand_mask(pred_mask)], axis=0)
	
	def add_query_and_bar(frame):
		""" Add GT/prediction bar into frame"""
		return np.concatenate([q_text, frame, bar], axis=0)

	final_clip = cc.fl_image(add_query_and_bar)
	
	if visualize:
		final_clip.ipython_display(maxduration=300)
	else:
		os.makedirs(f"visualization/{dt}/{policy}", exist_ok=True)
		save_to = f"visualization/{dt}/{policy}/{vid}.mp4"
		final_clip.write_videofile(save_to, fps=final_clip.fps)

def load_output(dt):
	if dt == "anet":
		from src.dataset import anet

		config_path = "pretrained_models/anet_LGI/config.yml"
		config = io_utils.load_yaml(config_path)["test_loader"]
		config["in_memory"] = False
		D = anet.ActivityNetCaptionsDataset(config)

		pred_path = "pretrained_models/anet_LGI/val_prediction.json"
		preds = io_utils.load_json(pred_path)
		vid_dir = "data/anet/raw_videos/validation/"
		
	elif dt == "charades":
		from src.dataset import charades

		config_path = "pretrained_models/charades_LGI/config.yml"
		config = io_utils.load_yaml(config_path)["test_loader"]
		config["in_memory"] = False
		D = charades.CharadesDataset(config)

		pred_path = "pretrained_models/charades_LGI/val_prediction.json"
		preds = io_utils.load_json(pred_path)
		vid_dir = "data/charades/raw_videos/"
		
	return D, preds, vid_dir

dt = "anet" # among anet|charades
D, preds, vid_dir = load_output(dt)

for i in range(1):
	try:
		make_result_video(preds, D, dt, vid_dir, "success", visualize=False)
	except:
		print("error occured :(")
		continue

for i in range(1):
	try:
		make_result_video(preds, D, dt, vid_dir, "failure", visualize=False)
	except:
		print("error occured :(")
		continue
		