import json
import argparse

from src.experiment import common_functions as cmf
from src.utils import io_utils


D = cmf.get_dataset("Charades")
dsets, L = cmf.get_loader(D, split=["test"],
                          loader_configs=[config["test_loader"]],
                          num_workers=params["num_workers"])