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

def l2norm(X):
	"""L2-normalize columns of X
	"""
	norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
	X = torch.div(X, norm)
	return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class EncoderVideoC3D(nn.Module):

	def __init__(self, img_dim, embed_size, num_layers=2, use_abs=False, no_imgnorm=False, use_bi=False):
		super(EncoderVideoC3D, self).__init__()
		self.embed_size = embed_size
		self.no_imgnorm = no_imgnorm
		self.use_abs = use_abs

		self.rnn = nn.GRU(img_dim, embed_size, num_layers, bidirectional=use_bi, batch_first=True,)
		if use_bi:
			self.fc = nn.Linear(2*embed_size, 2*embed_size)
		else:
			self.fc = nn.Linear(embed_size, embed_size)

		self.init_weights()

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
								  self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images):
		"""Extract image feature vectors."""
		# assuming that the precomputed features are already l2-normalized

		if len(images.shape) !=3:
			""" B X L X H, L is the number of images in video """
			images = images.unsqueeze(0)
		raw_features, _ = self.rnn(images)
		features = torch.sum(raw_features, dim=1) # Summing over number of images in video

		# normalize in the joint embedding space
		if not self.no_imgnorm:
			features = l2norm(features)

		# take the absolute value of embedding (used in order embeddings)
		if self.use_abs:
			features = torch.abs(features)

		out = self.fc(features)

		return out, raw_features 

	def load_state_dict(self, state_dict):
		"""Copies parameters. overwritting the default one to
		accept state_dict from Full model
		"""
		own_state = self.state_dict()
		new_state = OrderedDict()
		for name, param in state_dict.items():
			if name in own_state:
				new_state[name] = param

		super(EncoderVideoC3D, self).load_state_dict(new_state)

# RNN Based Language Model
class EncoderText(nn.Module):

	def __init__(self, vocab_size, word_dim, embed_size, num_layers=2,
				 use_abs=False, use_bi=False):
		super(EncoderText, self).__init__()
		self.use_abs = use_abs
		self.embed_size = embed_size

		# word embedding
		self.embed = nn.Embedding(vocab_size, word_dim)
		# query/description embedding
		self.use_bi = use_bi
		if use_bi:
			self.rnn = nn.GRU(word_dim, embed_size, num_layers, bidirectional=True, batch_first=True)
		else:
			self.rnn = nn.GRU(word_dim, embed_size, num_layers, bidirectional=False, batch_first=True)

		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)

	def forward(self, x, lengths):
		"""Handles variable size captions
		"""
		# Embed word ids to vectors
		x = self.embed(x)
		packed = pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)

		# Forward propagate RNN
		raw_out, _ = self.rnn(packed)

		# Reshape *final* output to (batch_size, hidden_size)
		padded = pad_packed_sequence(raw_out, batch_first=True)
		I = torch.LongTensor(lengths).view(-1, 1, 1)
		if self.use_bi: 
			if torch.cuda.is_available():
				I = Variable(I.expand(x.size(0), 1, 2*self.embed_size)-1).cuda()
			else:
				I = Variable(I.expand(x.size(0), 1, 2*self.embed_size)-1)
		else:
			if torch.cuda.is_available():
				I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
			else:
				I = Variable(I.expand(x.size(0), 1, self.embed_size)-1)
		out = torch.gather(padded[0], 1, I).squeeze(1)

		# normalization in the joint embedding space
		out = l2norm(out)

		# take absolute value, used by order embeddings
		if self.use_abs:
			out = torch.abs(out)

		return out, raw_out 
		

if __name__ == "__main__":
	imgs = torch.from_numpy(np.load("/Users/yanjungao/Desktop/LGI4temporalgrounding-master/data/charades/i3d_finetuned/3MSZA.npy")).squeeze(1).squeeze(1)
	enc = EncoderVideoC3D(1024, 1000,2)
	txt_enc = EncoderText(1000, 1000, 1000)
	txt_input = torch.tensor([1,2,3,4]).unsqueeze(0)
	lengths = torch.tensor(txt_input.shape[1]).unsqueeze(0)
	out = enc(imgs)
	txt_out = txt_enc(txt_input, lengths)

