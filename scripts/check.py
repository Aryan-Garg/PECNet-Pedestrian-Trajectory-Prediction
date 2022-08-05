import sys
sys.path.append("../utils/")
from social_utils import *
import yaml
from models import *
import numpy as np
from torch.utils.data import DataLoader
import argparse
# Args
parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='PECNET_social_model.pt')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()


with open("../config/" + args.config_filename, 'r') as file:
	try:
		hyper_params = yaml.load(file, Loader = yaml.FullLoader)
	except:
		hyper_params = yaml.load(file)
file.close()
# print(hyper_params)

train_file = np.load('../synthetic_datasets/train_dataset.npy', allow_pickle = True)

train_dataset = SocialDataset(set_name="train", b_size=hyper_params["train_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)
test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)

for traj in train_dataset.trajectory_batches:
	traj -= traj[:, :1, :]
	traj *= 1.86
for traj in test_dataset.trajectory_batches:
	traj -= traj[:, :1, :]
	traj *= 1.86

all_trajs1 = []
for batch in train_dataset.trajectory_batches:
	for traj in batch:
		all_trajs1.append(traj)

all_trajs2 = []
for batch in train_file:
	for traj in batch:
		all_trajs2.append(traj)


if np.array(all_trajs2).all() == np.array(all_trajs1).all():
	print(f"Good")