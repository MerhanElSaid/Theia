import torch
import torch.nn as nn
from model import *
from dataset import *


def Train(model, n_epochs, optimizer, criterion, Loaders, train_on_gpu):
    
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss = 0.0
        Train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        # model by default is set to train
        model.train()
        for batch_i, (data, target) in enumerate(Loaders['train']):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                target = target.type(torch.cuda.LongTensor)
            else:
                target = target.long()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss 
            train_loss += loss.item()

            if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
                print('Epoch %d, Batch %d loss: %.16f' %
                    (epoch, batch_i + 1, train_loss / 20))
                Train_loss += train_loss/20
                train_loss = 0.0


            #/////////////////////     Validation        \\\\\\\\\\\\\\\\\\\

        model.eval() # prep model for evaluation
        for eval_batch, (data, target) in enumerate(Loaders['valid']):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                target = target.type(torch.cuda.LongTensor)
            else:
                target = target.long()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss 
            valid_loss += loss.item()
            eval_batch += 1

        # print training loss per epoch
        print('Epoch %d, training loss: %.10f  validation loss : %.10f' %
            (epoch, Train_loss, valid_loss/eval_batch))
        Train_loss = 0.0
        valid_loss = 0.0
    return model

def main():
    Loaders = get_dataloaders()
    print('Data Preprocessed and got DataLoaders...')

    model = Face_Emotion_CNN()
    training_on_gpu = torch.cuda.is_available():
    
    epochs = 200
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('Starting Training loop...\n')
    model = Train(model, epochs, optimizer, criterion, Loaders, train_on_gpu)  


if __name__ == '__main__':
    main()