# !/usr/local/bin/python3
from net import *
from datafolder.folder import Train_Dataset
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import argparse
import matplotlib
from util import transfer_to_onehot

matplotlib.use('agg')  #
######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
	'market': 'Market-1501',
	'duke': 'DukeMTMC-reID',
}
model_dict = {
	'resnet18': ResNet18_nFC,
	'resnet34': ResNet34_nFC,
	'resnet50': ResNet50_nFC,
	'densenet': DenseNet121_nFC,
	'resnet50_softmax': ResNet50_nFC_softmax,
	'resnet50_attribute_baseline': ResNet50_attribute_baseline,
}

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data-path', default='../../../dataset',
                    type=str, help='path to the dataset')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--model', default='resnet50_softmax', type=str, help='model')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epoch', default=55, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=1, type=int, help='num_workers')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, args.model)

if not os.path.isdir(model_dir):
	os.makedirs(model_dir)


######################################################################
# Function
# --------


def save_network(network, epoch_label):
	save_filename = 'net_%s.pth' % epoch_label
	save_path = os.path.join(model_dir, save_filename)
	torch.save(network.cpu().state_dict(), save_path)
	if use_gpu:
		network.cuda()


######################################################################
# Draw Curve
# -----------
x_epoch = []
y_loss = {'train': [], 'val': []}  # loss history
y_err = {'train': [], 'val': []}

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
	x_epoch.append(current_epoch)
	ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
	ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
	ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
	ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
	if current_epoch == 0:
		ax0.legend()
		ax1.legend()
	fig.savefig(os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
image_datasets = {'train': Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                         train_val='train'),
                  'val': Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                       train_val='query')}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# images, indices, labels, ids, cams, names = next(iter(dataloaders['train']))

num_label = image_datasets['train'].num_label()
print('num_label' + str(num_label))
num_id = image_datasets['train'].num_id()
labels_list = image_datasets['train'].labels()

######################################################################
# Model and Optimizer
# ------------------
model = model_dict[args.model](num_label, num_id)  # for softmax
# model = model_dict[args.model](num_label)  #
if use_gpu:
	model = model.cuda()
# loss
# criterion_attr = nn.BCELoss()
# criterion_attr = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()  # for softmax loss
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                            weight_decay=5e-4, nesterov=True, )
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
	optimizer, step_size=50, gamma=0.1, )


#################
# util


######################################################################
# Training the model
# ------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs, apr_factor=6):
	since = time.time()
	
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		
		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode
			
			running_attr_loss = 0.0
			running_corrects = 0
			
			# Iterate over data.
			for count, data in enumerate(dataloaders[phase]):
				# get the inputs
				images, indices, labels, ids, cams, names = data
				# wrap them in Variable
				if use_gpu:
					images = images.cuda()
					labels = labels.t().cuda()  # 32 * 30
					indices = indices.cuda()
					ids = ids.cuda()
				# print(ids)
				images = images
				# labels = labels.float()  # expected float for target in BCELoss()
				indices = indices
				
				# zero the parameter gradients
				optimizer.zero_grad()
				
				# forward
				# outputs, _ = model(images)
				outputs, _ = model(images)  # return
				
				label_output = outputs[:-1]  # 30 * 32 * 2
				id_output = outputs[-1]
				# ======== attribute loss and classification === #
				# attr_loss = criterion_attr(label_output[0], onehot_label[0])  # classifier for attribute 0
				# for i in range(1, len(labels)):
				# 	attr_loss += criterion_attr(label_output[i], onehot_label[i])
				
				# labels = transfer_to_onehot(labels)
				# labels = labels.t()  # M * batch_size
				_, pred = torch.max(id_output.data, 1)
				id_loss = criterion(id_output, indices)
				
				attr_loss = criterion(label_output[0], labels[0])
				for i in range(1, len(labels)):
					attr_loss += criterion(label_output[i], labels[i])
				
				APR_loss = apr_factor * id_loss + attr_loss / len(labels)  # attr_loss already averaged
				
				# backward + optimize only if in training phase
				if phase == 'train':
					APR_loss.backward()
					optimizer.step()
				
				# attr_preds = torch.gt(
				# 	label_output, torch.ones_like(label_output) / 2).data
				
				# statistics
				running_attr_loss += APR_loss.item()
				# running_corrects += torch.sum(attr_preds ==
				#                               labels.data.byte()).item() / num_label
				running_corrects += float(torch.sum(pred == indices.data))
				print('step : ({}/{})  |  loss : {:.4f}'.format(count *
				                                                args.batch_size, dataset_sizes[phase],
				                                                attr_loss.item()))
			
			epoch_loss = running_attr_loss / len(dataloaders[phase])
			epoch_acc = running_corrects / dataset_sizes[phase]
			
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			y_loss[phase].append(epoch_loss)
			y_err[phase].append(1.0 - epoch_acc)
			# deep copy the model
			if phase == 'val':
				last_model_wts = model.state_dict()
				if epoch % 10 == 9:
					save_network(model, epoch)
				draw_curve(epoch)
	
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	
	# load best model weights
	model.load_state_dict(last_model_wts)
	save_network(model, 'last')


######################################################################
# Main
# -----
model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                    num_epochs=args.num_epoch)
