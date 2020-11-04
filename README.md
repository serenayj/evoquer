Temporal Grounding through Video Pivoted Machine Translation
(VPMT)

# Environment Setup 
参考 LGI (src/anaconda_environment.md)

# Folder and Files 
- data : 
	folder for data. Download from src/scripts/prepare_data.sh 
	- charades 
	- activitynet (TBD)
- src: 
	Yanjun modified version of LGI model (orig from (https://github.com/JonghwanMun/LGI4temporalgrounding))
	- model 
		- building_blocks.py: network modules, LGI loss functions  
		- abstract_networks.py: abstract methods for LGI, including maintaining best scores record, print_status (for LGI only) 
		- LGI.py: network setting for LGI model, inherited class from abstract_networks.py, this function is really important here: 
			- prepare_batch_w_pipeline: sampling batch data for pipeline input 
	- dataset: folder of dataset loader 
		- abstract_dataset.py : abtract methods for data loader 
		- anet.py: activitynet data loader (TBD) 
		- charades.py: Charades dataloader 
	- utils : utility functions 
		- eval_utils.py: evaluation codes
	- experiment
		- common_functions.py: function test is for evaluation 
- seq2seq : 
	IBM seq2seq package for translation (orig from (https://github.com/IBM/pytorch-seq2seq)) 
	- models
	 	- DecoderRNN.py: RNN modules for decoder 
- VSE : 
	visual embedding source codes from (https://github.com/fartashf/vsepp), re-oraganized into vse_video_enc.py  
- ymls:
	containing config file (.ymls) as config input to LGI model 


# VPMT Pipeline python files 
- trainer.py:
	supervised trainer启动代码，包括data loader, train, validate, save checkpoint  
- VPMT.py:
	VPMT pipeline with LGI, VSE, Translation, and miscellaneous functions (optimizer, forward, update)
- VPCLS.py: 
	VPMT pipeline with action and object classifcation instead of translation 
- vse_video_enc.py: 
	Visual embedding modules 
- vpmt_config.py 
	configuration 

# 启动实例 
``python trainer.py >> log_*setting*.py
``


