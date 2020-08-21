from vpmt_config import * 
from trainer import SupervisedTrainer 
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
	"batch_size": 1,
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
	bot.prefix += "_SGD_steplr_"
	bot.train()