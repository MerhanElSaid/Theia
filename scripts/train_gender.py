import sys
from pathlib import Path
project_dir = (Path(__file__).parent / '..').resolve()
if str(project_dir) not in sys.path: sys.path.insert(0, str(project_dir))

import torch 
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
torch.manual_seed(0)

from models.gender import loadModel
from utils import check_acc, plot_performance_curves, save_checkpoint, MyDataset

use_gpu = torch.cuda.is_available()
num_epochs = 100
batch_size = 128

def main():
	MAIN_DIR = project_dir._str
	
	train_transform = transforms.Compose([
		transforms.Resize(226),
		transforms.RandomCrop(224),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(10),
		transforms.Grayscale(num_output_channels=1),
		transforms.ToTensor(),
		transforms.Normalize([0.485],[0.229])
		])

	test_transform = transforms.Compose([
		transforms.Resize(226),
		transforms.CenterCrop(224),
		transforms.Grayscale(num_output_channels=1),
		transforms.ToTensor(),
		transforms.Normalize([0.485],[0.229])
		])

	all_dataset = dataset.ImageFolder(root=MAIN_DIR +'/dataset/gender')
	lengths = [int(len(all_dataset)*0.9), int(len(all_dataset)*0.1)+1]
	subsetA, subsetB = torch.utils.data.random_split(all_dataset, lengths)

	train_data = MyDataset(subsetA, transform=train_transform)
	test_data = MyDataset(subsetB, transform=test_transform)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=0, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False, num_workers=0, pin_memory=True)


	myModel = loadModel()

	myModel.load_state_dict(torch.load(MAIN_DIR+'/checkpoints/gender/deploy_80_model.pth.tar')['state_dict'])

	for name, param in myModel.named_parameters():
		if("bn" not in name):
			param.requires_grad = False
	
	if use_gpu:
		myModel.cuda()

	
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(myModel.parameters(),lr=0.001)

	train_acc_history = []
	val_acc_history = []
	epoch_history = []
	best_val_acc = 0.0


	for epoch in range(num_epochs):
		
		print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
		for i,(images,labels) in enumerate(train_loader):
			optimizer.zero_grad()
			images = Variable(images).repeat(1, 3, 1, 1)
			labels = Variable(labels)
			if use_gpu:
				images,labels = images.cuda(),labels.cuda()
			
			pred_labels = myModel(images)
			loss = criterion(pred_labels,labels)
			loss.backward()
			optimizer.step()

			if (i+1) % 5 == 0:
				print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
					%(epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.item()))

		if epoch % 10 ==0 or epoch == num_epochs-1:

			myModel.eval()
			train_acc = check_acc(myModel, train_loader, use_gpu)
			train_acc_history.append(train_acc)
			print('Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc))

			val_acc = check_acc(myModel, test_loader, use_gpu)
			myModel.train()
			val_acc_history.append(val_acc)
			print('Validation accuracy for epoch {} : {} '.format(epoch + 1,val_acc))
			epoch_history.append(epoch+1)
			plot_performance_curves(train_acc_history, val_acc_history, epoch_history, MAIN_DIR+'/logs/gender/acc_recode.png')

			is_best = val_acc > best_val_acc
			best_val_acc = max(val_acc,best_val_acc)
			save_checkpoint(
				{'epoch':epoch+1,
				'state_dict':myModel.state_dict(),
				'best_val_acc':best_val_acc,
				'optimizer':optimizer.state_dict()},is_best, MAIN_DIR+'/checkpoints/gender/best_checkpoint.pth.tar')

if __name__ == '__main__':
	main()