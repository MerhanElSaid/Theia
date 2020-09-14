import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def check_acc(cnn, data_loader, use_gpu):
	num_correct,num_sample = 0, 0
	for images,labels in data_loader:
		images = Variable(images).repeat(1, 3, 1, 1)
		labels = Variable(labels)
		if use_gpu:
			images,labels = images.cuda(),labels.cuda()
		outputs = cnn(images)
		_,pred = torch.max(outputs.data,1)
		num_sample += labels.size(0)
		num_correct += (pred == labels).sum()
	return float(num_correct)/num_sample

def plot_performance_curves(train_acc_history, val_acc_history, epoch_history, save_path):
	plt.figure()
	plt.plot(np.array(epoch_history),np.array(train_acc_history),label = 'Training accuracy')
	plt.plot(np.array(epoch_history),np.array(val_acc_history),label = 'Validation accuracy')
	plt.title('Accuracy on training and validation')
	plt.ylabel('Accuracy')
	plt.xlabel('Number of epochs')
	plt.legend()
	plt.savefig(save_path)

def save_checkpoint(state,is_best,file_name = 'checkpoints/checkpoint.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'checkpoints/model_best.pth.tar')