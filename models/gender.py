import torch 
import torch.nn as nn
import torch.nn.functional as F

def loadModel(): 
	myModel = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
	myModel.fc = nn.Sequential(nn.Linear(myModel.fc.in_features,512), nn.ReLU(), nn.Dropout(), nn.Linear(512, 2))
	return myModel