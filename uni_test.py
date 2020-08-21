"""
Walk thru LGI process
"""

import sys
sys.path.append("/Users/yanjungao/Desktop/VPMT/")
from src.utils import io_utils, eval_utils
config_path="/Users/yanjungao/Desktop/LGI4temporalgrounding-master/pretrained_models/charades_LGI/config.yml"
full_config= io_utils.load_yaml(config_path)
config = io_utils.load_yaml(config_path)["train_loader"]
from src.dataset.charades import * 
D = CharadesDataset(config)
vis_data = D.get_samples(int(10))
#net_inps, gts = model.prepare_batch(batch)

"""
dset, l = get_loader()

from src.dataset import anet, charades_two_streams
from src.model import building_networks as bn
from src.utils import utils, io_utils
def get_method(method_type):
	if method_type.startswith("tgn"):
		M = bn.get_temporal_grounding_network(None, method_type, True)
	else:
		raise NotImplementedError("Not supported model type ({})".format(method_type))
	return M
item = D.get_samples(1)
batch = {k: [d[k] for d in item[0]] for k in item[0][0].keys()} 

from VPMT import VPMT
M = get_method("tgn_lgi")
model = M(full_config)

# for batch in l["train"]:
# 	print(type(batch))
	#net_inps, gts = model.prepare_batch(batch)

for batch in l["test"]:
	batch 
"""


from src.dataset import anet, charades_two_streams
from src.model import building_networks as bn
from src.utils import utils, io_utils
def get_method(method_type):
	if method_type.startswith("tgn"):
		M = bn.get_temporal_grounding_network(None, method_type, True)
	else:
		raise NotImplementedError("Not supported model type ({})".format(method_type))
	return M

from VPMT import VPMT
M = get_method("tgn_lgi")
#config_path ="/Users/yanjungao/Desktop/LGI4temporalgrounding-master/src/experiment/options/charades/tgn_lgi/LGI.yml"
#config = io_utils.load_yaml(config_path)
model = M(full_config)
# L = {} 
# L["test"] = D
# from src.dataset.charades import create_loaders 
# vis_data = D.get_samples(int(10))
# vis_data[0][0].keys() 
# batch = L['test'].collate_fn(vis_data)
net_inps, gts = model.prepare_batch(vis_data)
out = model(net_inps)
