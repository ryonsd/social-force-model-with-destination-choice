import numpy as np
import math
import pandas as pd
from statistics import mean, stdev
from tqdm.notebook import trange, tqdm
import warnings
import argparse
import torch
import time
from tqdm import tqdm
import yaml
import os
warnings.simplefilter('ignore')

from SFM import make_sfm
import sys
sys.path.append("../src")

from main import Agent, run
# from make_sfm import run_sfm
from social_force_model import SocialForceModel
from destination_choice_model import DestinationChoiceModel

#yamlから読み込むデータパスを指定
#データパスを使ってmake_sfmを実行

if __name__ == '__main__':
    file_path = 'config.yaml'
    with open(file_path) as f:
        config = yaml.safe_load(f)
    data_path = config['input']
    save_path = config['output']
    device = config['device']
    for i in range(len(data_path)):
        dp = data_path[i]
        sp = save_path[i]
        command = 'python SFM.py -i ' + dp + ' -o ' + sp + ' -d ' + device
        print(command)
        command = command.strip() + '\n'
        run = os.system(command)
        print('command number:', i)
        if run == 0:
            print('success')
        else:
            print('failed')
    print('finish')
    