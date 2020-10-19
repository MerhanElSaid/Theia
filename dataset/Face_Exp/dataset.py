import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import csv 

 
def prepareData(file, X, Y, b):
  with open(file,'r') as csvin:
    data=csv.reader(csvin)
    for row in data:
        temp = torch.zeros((48, 48), dtype= int)
        if row[b] == 'pixels':
          continue
        pixs = row[b].split( )
        for j in range(48):
          for i in range(48):
              temp[i][j] = (float(pixs[i+(j*48)]))
        
        if b:
          Y.append(int(row[0]))
        X.append(temp)
  return X, Y


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, x_data, y_data, transform=None):
        self.transform = transform
        self.x_data = x_data
        self.y_data = y_data
 
    def __getitem__(self, index):
        x = self.x_data[index]
        x = x.numpy()
        x = Image.fromarray(np.uint8(x))
        y = self.y_data[index]
        if self.transform:
          x = self.transform(x)
             
        return x, y   


def get_dataloaders(path_csv='', train_batch=3000, val_batch=500):
  Training_x = []
  Training_y = []

  Training_x, Training_y = prepareData(path_csv, Training_x, Training_y, 1) 


  train_transform = transforms.Compose([transforms.RandomCrop(44),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor()
                                      ])
  dataset = CustomTensorDataset(Training_x, Training_y, transform= train_transform)
  test = int(len(dataset)*0.2)  
  valid = int(len(dataset)*0.1)

  test_data = torch.utils.data.Subset(dataset, range(test)) 

  valid_data = torch.utils.data.Subset(dataset, range(test, test+valid))

  train_data = torch.utils.data.Subset(dataset, range(valid + test, len(dataset)))


  train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch, shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = val_batch)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size = val_batch)

  Loaders ={
      'train' : train_loader,
      'valid' : valid_loader,
      'test' : test_loader
  }
  return Loaders
