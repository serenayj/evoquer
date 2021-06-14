from vpmt_config import * 
from trainer import SupervisedTrainer 
from src.experiment import common_functions as cmf

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
	"batch_size": 100,
	}
	import sys
	#sys.path.append("/Users/yanjungao/Desktop/VPMT/")
	from src.utils import io_utils, eval_utils
	#config_path="/Users/yanjungao/Desktop/LGI4temporalgrounding-master/pretrained_models/charades_LGI/config.yml"
	config_path="ymls/config.yml"
	full_config= io_utils.load_yaml(config_path)
	config = io_utils.load_yaml(config_path)["train_loader"]
	from src.dataset.charades import * 
	train_D = CharadesDataset(config)
	config = io_utils.load_yaml(config_path)["test_loader"]
	test_D = CharadesDataset(config)
	m_config = model_args(full_config, pip_config) # this has to be full model 
	bot = SupervisedTrainer(m_config,m_config.lgi_arg, config, None, test_D)
	bot.model.LGI_model.device = torch.device("cpu")
	#bot.train()
	#pretrain_dict = torch.load()
	bot.model.load_state_dict(torch.load("60_step_simp_trans_24f_nonllavg__vpmt.pkl",map_location="cpu"))
	#bot.model.load_state_dict(torch.load("results/charades/tgn_lgi/LGI_2020-07-20/checkpoints/best.pkl",map_location="cpu"))
	#bot.model.LGI_model.load_state_dict(torch.load("results/charades/tgn_lgi/LGI_2020-07-20/checkpoints/best.pkl",map_location="cpu"))
	#eval_logger = cmf.create_logger(bot.lgi_config, "EPOCH", "eval_scores_simpl_trans.log")
	#bot.validate_LGI(eval_logger, config, 30)
	if bot.model.simplified_trans:
		eval_logger = "eval_scores_simpl_trans_nonllavg.log"
		itow = {int(k): v for k,v in bot.model.idx_vocab.items()} 
	else:
		itow = {v:k for k,v in test_D.wtoi.items()}
	bot.validate_translate(eval_logger, config, 30)
	bot.model.translate(itow, bot.gts,write=True) 


