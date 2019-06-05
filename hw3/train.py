# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
import sys

if len(sys.argv) < 5:
    sys.stderr.write("Wrong Usage: %s %s %s %s %s\n" 
        % (sys.argv[0], "LAYERS", "KERNEL_SIZE", "CHANNELS", "LEARNING_RATE"))
    exit(-1)

batch_size = 16
max_num_epoch = 100
layers = int(sys.argv[1])
kernel_size = int(sys.argv[2])
channel = int(sys.argv[3])
lr = float(sys.argv[4])
padding = int(kernel_size/2)
MODEL_DIR = 'models'
model_name = "%d_%d_%d_%s" % (layers, kernel_size, channel, sys.argv[4])
BATCH_NORMALIZATION = False
TANH = True

if BATCH_NORMALIZATION:
    model_name += "_bn"
if TANH:
    model_name += "_tanh"

# ---- options ----
DEVICE_ID = 'cuda' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False

# --- imports ---
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
torch.multiprocessing.set_start_method('spawn', force=True)
# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset' 
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

def margin12Error(inputs, targets):
    inp = (inputs.reshape(1,-1).squeeze() + 1.0)*255.0
    trg = (inputs.reshape(1,-1).squeeze() + 1.0)*255.0
    err = (torch.abs(inp - trg) < 12).sum() / len(inputs)
    return err

# ---- ConvNets -----
class Net1Layer(nn.Module):
    def __init__(self):
        super(Net1Layer, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size, padding=padding)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = self.conv1(grayscale_image)
        if TANH:
            x = F.tanh(x)
        return x

class Net1LayerBN(nn.Module):
    def __init__(self):
        super(Net1LayerBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(3)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = self.conv1(grayscale_image)
        x = self.bn1(x)
        if TANH:
            x = F.tanh(x)
        return x

class Net2Layer(nn.Module):
    def __init__(self):
        super(Net2Layer, self).__init__()
        self.conv1 = nn.Conv2d(1, channel, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channel, 3, kernel_size, padding=padding)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = F.relu(self.conv1(grayscale_image))
        x = self.conv2(x)
        if TANH:
            x = F.tanh(x)
        return x

class Net2LayerBN(nn.Module):
    def __init__(self):
        super(Net2LayerBN, self).__init__()
        self.conv1 = nn.Conv2d(1, channel, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, 3, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = F.relu(self.conv1(grayscale_image))
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if TANH:
            x = F.tanh(x)
        return x

class Net4Layer(nn.Module):
    def __init__(self):
        super(Net4Layer, self).__init__()
        self.conv1 = nn.Conv2d(1, channel, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, padding=padding)  
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, padding=padding)  
        self.conv4 = nn.Conv2d(channel, 3, kernel_size, padding=padding)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = F.relu(self.conv1(grayscale_image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        if TANH:
            x = F.tanh(x)
        return x

class Net4LayerBN(nn.Module):
    def __init__(self):
        super(Net4LayerBN, self).__init__()
        self.conv1 = nn.Conv2d(1, channel, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, padding=padding)  
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, padding=padding) 
        self.bn3 = nn.BatchNorm2d(channel) 
        self.conv4 = nn.Conv2d(channel, 3, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = self.conv1(grayscale_image)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = self.bn4(x)
        if TANH:
            x = F.tanh(x)
        return x

# ---- training code -----
device = torch.device(DEVICE_ID)
print('device: ' + str(device))
if layers == 1:
    if BATCH_NORMALIZATION:
        net = Net1LayerBN().to(device=device)
    else:
        net = Net1Layer().to(device=device)
elif layers == 2:
    if BATCH_NORMALIZATION:
        net = Net2LayerBN().to(device=device)
    else:
        net = Net2Layer().to(device=device)
elif layers == 4:
    if BATCH_NORMALIZATION:
        net = Net4LayerBN().to(device=device)
    else:
        net = Net4Layer().to(device=device)
else:
    sys.stderr.write("UNSUPPORTED LAYER COUNT: %d\n" % (layers))
    exit(-1)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
train_loader, val_loader = get_loaders(batch_size,device)

if LOAD_CHKPT:
    print('loading the model from the checkpoint')
    net.load_state_dict(os.path.join(LOG_DIR,'checkpoint.pt'))

best_val_loss = float('Inf')
best_epoch = -1

print('training begins')
for epoch in range(max_num_epoch):  
    running_loss = 0.0 # training loss of the network
    for iteri, data in enumerate(train_loader, 0):
        inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

        optimizer.zero_grad() # zero the parameter gradients

        # do forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.item()
        # print_n = 100 # feel free to change this constant
        # if iteri % print_n == (print_n-1):    # print every print_n mini-batches
        #     print('[%d, %5d] network-loss: %.3f' %
        #           (epoch + 1, iteri + 1, running_loss / 100))
        #     running_loss = 0.0
        #     # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

        #if (iteri==0) and VISUALIZE: 
        #    hw3utils.visualize_batch(inputs,preds,targets)
    running_loss = running_loss / len(train_loader)

    if True: #epoch % 5 == 4:
        with torch.no_grad():
            val_loss = 0.0 # validation loss of the network
            margin_error = 0.0
            for data in val_loader:
                inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

                # do forward, get validation loss
                preds = net(inputs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                #v1 = torch.abs(torch.sub(preds, targets) > 12.0/255.0).sum().to(dtype=torch.float)
                #v2 = float(preds.reshape(-1).shape[0])
                margin_error +=  torch.abs(torch.sub(preds, targets) > 12.0/255.0).sum().to(dtype=torch.float) / float(preds.reshape(-1).shape[0])


            val_loss = val_loss / len(val_loader)
            margin_error = margin_error / len(val_loader)
            if (val_loss <= best_val_loss):
                best_val_loss = val_loss
                best_epoch = epoch + 1
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                torch.save(net.state_dict(), os.path.join(MODEL_DIR,'conv_%s.model' % (model_name)))
            elif (val_loss > best_val_loss*3):
                print('Force end of training at epoch: %d (%f|%f)' % (epoch+1, val_loss, best_val_loss))
                break
            print('[%3d]\tTrain-loss: %f\tValidation-loss: %f\t12Margin-error: %f' % (epoch+1, running_loss, val_loss, margin_error))
            #print('Validation-loss: %f' % (val_loss))
            #print('12Margin-error: %f' % (margin_error))

    #print('Saving the model, end of epoch %d with train-loss: %f' % (epoch+1, running_loss))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save(net.state_dict(), os.path.join(LOG_DIR,'model_%s.pt' % (model_name)))
    hw3utils.visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR,'model_%s.png' % (model_name)))

print('Finished Training with Minimum Validation Loss: %f at Epoch: %d' % (best_val_loss, best_epoch + 1))


