import torch


def transfer_to_onehot(label):
	"""
	:param label: 30*32*2
	:return: should be 30*32(batch_size)*2, label for every attr in each batch
	"""
	batch_size = label.shape[0]
	class_num = label.shape[1]
	# label = torch.Tensor(label).t()  # 30*32 the output of each attr in 32(batch_size)
	onehot = torch.zeros(class_num, batch_size, dtype=torch.long)
	for i in range(batch_size):
		for j in range(class_num):
			if label[i][j].cpu().numpy() != 0:
				onehot[j][i] = 1
			else:
				onehot[j][i] = 0
			print(str(onehot[j][i].data.cpu().numpy()) + " ", end=' ')
		print()
	return onehot.cuda()


# print(123)

#
# test_label = torch.tensor(torch.rand(32,30)*10)
# print(transfer_to_onehot(test_label))
