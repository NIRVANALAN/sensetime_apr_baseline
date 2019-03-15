import torch


def transfer_to_onehot(label):
    onehot = torch.zeros([len(label),2])
    for i in range(len(label)):
        if label[i]:
            onehot[i][0] = 1
        else:
            onehot[i][1] = 1
    return onehot

# print(123)


label = torch.tensor([1, 0, 0, 1, 1])
print(transfer_to_onehot(label))
