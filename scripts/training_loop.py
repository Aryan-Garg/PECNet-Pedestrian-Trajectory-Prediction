import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
sys.path.append("../utils/")
from social_utils import *
import yaml
from models import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

import numpy as np
import matplotlib.pyplot as plt
import random
import time 

# Args
parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='PECNET_social_model.pt')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()

# Device
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)

# Config File stuff
with open("../config/" + args.config_filename, 'r') as file:
	try:
		hyper_params = yaml.load(file, Loader = yaml.FullLoader)
	except:
		hyper_params = yaml.load(file)
file.close()
# print(hyper_params)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def train(train_dataset, syn_ds = "", decouple = False, epochNum = -1):
	model.train()
	train_loss = 0
	total_rcl, total_kld, total_adl = 0, 0, 0
	criterion = nn.MSELoss()

	if syn_ds != "":
		data_syn = np.load(f'../synthetic_datasets/{syn_ds}.npy', allow_pickle=True)
	else:
		data_syn = np.load('../synthetic_datasets/train_dataset.npy', allow_pickle=True)

	# cnt = 0
	for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
		mask2 = np.zeros((traj.shape[0], traj.shape[0]))
		traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
	
	# for batch in data_syn:
		
		# traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask2).to(device), torch.DoubleTensor(initial_pos).to(device)
		
		# print(traj.shape, mask2.shape, initial_pos.shape)
		# print(data_syn[0].shape, mask2.shape, data_syn[0][:,0,:].shape)
		# batch = np.array(batch)
		
		# traj, mask, initial_pos = torch.DoubleTensor(batch).to(device), torch.DoubleTensor(mask2).to(device), torch.DoubleTensor(batch[:,7,:]).to(device)
		
		x = traj[:, :hyper_params['past_length'], :]
		y = traj[:, hyper_params['past_length']:, :]

		x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
		x = x.to(device)
		dest = y[:, -1, :].to(device)
		future = y[:, :-1, :].contiguous().view(y.size(0),-1).to(device)

		dest_recon, mu, var, interpolated_future = model.forward(x, initial_pos, dest=dest, mask=mask, device=device)

		optimizer.zero_grad()

		if decouple:
			rcl, kld = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future, decouple)
			loss = rcl + kld*hyper_params["kld_reg"]
		else:
			rcl, kld, adl = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future, decouple)
			loss = rcl + kld*hyper_params["kld_reg"] + adl*hyper_params["adl_reg"]
		
		# if cnt == len(data_syn)-2:
		# 	print(f"Total loss {loss.item():0.6f}")
		# 	img_grad = gradient(dest_recon, dest)
		# 	img_laplacian = laplace(dest_recon, dest)

		# 	fig, axes = plt.subplots(1,3, figsize=(18,6))
		# 	axes[0].imshow(model_output.cpu().view(256,256).detach().numpy())
		# 	axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256,256).detach().numpy())
		# 	axes[2].imshow(img_laplacian.cpu().view(256,256).detach().numpy())
		# 	plt.savefig(f'../Logs_LossPlots/{epochNum}_{cnt}')

		# cnt += 1

		loss.backward()

		train_loss += loss.item()
		total_rcl += rcl.item()
		total_kld += kld.item()
		if not decouple:
			total_adl += adl.item()
		optimizer.step()

		# # For ENC_PAST Viz.
		# break 
		# # Remove afterwards ^^^

	return train_loss, total_rcl, total_kld, total_adl


def test(test_dataset, dsTest = "", best_of_n = 1):
	'''Evalutes test metrics. Assumes all test data is in one batch'''

	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int

	# if dsTest == "":
	# 	data_syn = np.load('../synthetic_datasets/test_dataset.npy', allow_pickle=True)
	# else:
	# 	data_syn = np.load(f'../synthetic_datasets/v4/{dsTest}.npy', allow_pickle=True)


	with torch.no_grad():
		# for batch in data_syn:
		# 	mask2 = np.zeros((batch.shape[0], batch.shape[0]))
		# 	# print(traj.shape, mask2.shape, initial_pos.shape)
		# 	# print(data_syn[0].shape, mask2.shape, data_syn[0][:,0,:].shape)
		# 	batch = np.array(batch)
		
		# 	batch -= batch[:, :1, :] # Shifting to Origin; Just in case 
		# 	traj, mask, initial_pos = torch.DoubleTensor(batch).to(device), torch.DoubleTensor(mask2).to(device), torch.DoubleTensor(batch[:,7,:]).to(device)
		
		for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
			traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
			x = traj[:, :hyper_params['past_length'], :]
			y = traj[:, hyper_params['past_length']:, :]
			y = y.cpu().numpy()

			# reshape the data
			x = x.view(-1, x.shape[1]*x.shape[2])
			x = x.to(device)

			dest = y[:, -1, :]
			all_l2_errors_dest = []
			all_guesses = []
			for _ in range(best_of_n):

				dest_recon = model.forward(x, initial_pos, device=device)
				dest_recon = dest_recon.cpu().numpy()
				all_guesses.append(dest_recon)

				l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)
				all_l2_errors_dest.append(l2error_sample)

			all_l2_errors_dest = np.array(all_l2_errors_dest)
			all_guesses = np.array(all_guesses)
			# average error
			l2error_avg_dest = np.mean(all_l2_errors_dest)

			# choosing the best guess
			indices = np.argmin(all_l2_errors_dest, axis = 0)

			best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]

			# taking the minimum error out of all guess
			l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))

			best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

			# using the best guess for interpolation
			interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
			interpolated_future = interpolated_future.cpu().numpy()
			best_guess_dest = best_guess_dest.cpu().numpy()

			# final overall prediction
			predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
			predicted_future = np.reshape(predicted_future, (-1, hyper_params['future_length'], 2)) # making sure
			# ADE error
			l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis = 2))

			l2error_overall /= hyper_params["data_scale"]
			l2error_dest /= hyper_params["data_scale"]
			l2error_avg_dest /= hyper_params["data_scale"]

			# print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
			# print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

	return l2error_overall, l2error_dest, l2error_avg_dest


# Note start time
rawTime_start = time.time()
mega_startTime = time.localtime(rawTime_start)

# LOG Filename
LOG_FNAME2 = f'../Logs_LossPlots/LOG_SAME_DS_elaborate.txt'

# Log starting time and decoupling info
mfile2 = open(LOG_FNAME2, 'a')
print(f"<<<<<<<<<< Full Experiment(logged to:{LOG_FNAME2})\n Started at {time.asctime(mega_startTime)} >>>>>>>>>>", file=mfile2)
mfile2.close()

# File name LR correspondence:  
#             0       1       2       3       4            
adam_lrs = [0.001, 0.0005, 0.0003, 0.0001]

# Their dataset loading, creation...
train_dataset = SocialDataset(set_name="train", b_size=hyper_params["train_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)
test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)

# shift origin and scale data
for traj in train_dataset.trajectory_batches:
	traj -= traj[:, :1, :]
	traj *= hyper_params['data_scale']
for traj in test_dataset.trajectory_batches:
	traj -= traj[:, :1, :]
	traj *= hyper_params['data_scale']

epochs = 1000
mfile = open(LOG_FNAME2, 'a')

v4_train = ["test_dataset"]

# for i in range(10, 101, 5):
# 	v4_train.append(f"v4_{i}.npy")


# print(f"PURE SYN Train Dataset Names: {v4_train}")
v4_test = [""]

for ds in v4_train:
	print(f"Dataset: test(2829)\n",file=mfile)
	print(f"Dataset: {ds[:-4]}")

	for i in range(len(adam_lrs)):
		pltSaveFig = f'../Logs_LossPlots/SAME_DS_elaborate_ALR{i}'
		pltTitle = f'SAME DS elaborate ALR_{i}'

		print(f"ADAM LR: {adam_lrs[i]}\nEpochs: {epochs}")
		print(f"ADAM LR: {adam_lrs[i]}\nEpochs: {epochs}", file=mfile)

		model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
		model = model.double().to(device)
		optimizer = optim.Adam(model.parameters(), lr= adam_lrs[i])
	
		best_test_loss = 100 # start saving after this threshold
		best_endpoint_loss = 100
		N = hyper_params["n_values"]

		ADE_bl_List = []
		FDE_bl_List = []
		
		for e in range(epochs):
			if (e+1) % (epochs//5) == 0:
				print(f"Checkpoint: {int((e+1)/(epochs//5))}")

			train_loss, rcl, kld, adl = train(train_dataset, syn_ds = ds, epochNum = e)
			test_loss, final_point_loss_best, final_point_loss_avg = test(test_dataset, v4_test[0], best_of_n = N)

			FDE_bl_List.append(final_point_loss_best)
			ADE_bl_List.append(test_loss)
			
			if (e+1) % 50 == 0:
				print(f"\n{e+1}/{epochs}",file=mfile)
				print("Train Loss", train_loss, file=mfile)
				print(f"RCL: {rcl:.3f} | KLD: {kld:.3f} | ADL: {adl:.3f}", file=mfile)

			if final_point_loss_best < best_endpoint_loss:
				best_endpoint_loss = final_point_loss_best
				print(f'##### Epoch {e+1} | BEST FDE {best_endpoint_loss:0.2f} #####', file = mfile)
	
			if best_test_loss > test_loss:
				print(f'################## Epoch {e+1} | BEST ADE {test_loss:0.2f} ##################', file = mfile)
				best_test_loss = test_loss
				save_path = f'../saved_models/SAME_DS_elaborate_ALR{i}.pt'
				torch.save({
							'hyper_params': hyper_params,
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'best FDE': best_endpoint_loss,
							'best ADE': best_test_loss
							}, save_path)
					# print("Saved model to:\n{}".format(save_path))
		
			

			# print(f"{e+1}/{epochs}:\nTest ADE: {test_loss:.2f} | Test Avg. FDE: {final_point_loss_avg:.2f} | Test Min FDE: {final_point_loss_best:.2f}")
			# print("---------------")
			# print("Test Best ADE Loss So Far (N = {})".format(N), best_test_loss)
			# print("Test Best Min FDE (N = {})".format(N), best_endpoint_loss)
			# print("---------------")

		try:
			figfig = plt.figure(figsize=(16,12))
			plt.yticks(np.arange(0,101,5))
			plt.plot(np.arange(len(FDE_bl_List)), FDE_bl_List, label='FDE Loss', color='orange')
			plt.plot(np.arange(len(ADE_bl_List)), ADE_bl_List, label='ADE Loss', color='blue')
			plt.grid(True)
			plt.legend()
			plt.title(pltTitle)
			plt.savefig(pltSaveFig)

		except Exception as inst:
			print(f"\n[!]Couldn't plot loss curves",file=mfile)
			print(type(inst), file=mfile)
			print(inst.args, file=mfile)
			print(inst, file=mfile) 
			print("~~~~~~~~~~~~~~~~~~~~~", file=mfile)
            
mfile.close()

# (Folded)End Time Stuff
rawTime_end = time.time()
mega_endTime = time.localtime(rawTime_end)

# Log end and elaspsed times as well
# mfile = open(LOG_FNAME, 'a')
# print(f"<<<<<<<<<< Full Experiment Ended at {time.asctime(mega_startTime)} >>>>>>>>>>", file=mfile)
# print(f"\nTotal Elapsed Time: {time.asctime( time.localtime( rawTime_end - rawTime_start ) ) }", file=mfile)
# mfile.close()

mfile2 = open(LOG_FNAME2, 'a')
print(f"<<<<<<<<<< Full Experiment Ended at {time.asctime(mega_startTime)} >>>>>>>>>>", file=mfile2)
print(f"\nTotal Elapsed Time: {time.asctime( time.localtime( rawTime_end - rawTime_start ) ) }", file=mfile2)
mfile2.close()