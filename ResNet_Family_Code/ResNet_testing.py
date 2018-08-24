# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
ResNet34 Classifier
"""
"""
Author: jianan liu
Version: 1.0.0
"""
"-------------------------------------------------------------------------------------------------"
"""
This is a demo code of ResNet.
    1) ResNet34[1] in theory, works better than LeNet, especially on the following items:
        a) avoiding the "gradient vanish" problem when using very deep neural network(more than hundreds layers) by adding "identity short cut"(residual link)
        b) even might make the very non-convex shape of very deep neural network(without identity short cut) which may have a lot of saddle points
            becoming much more convex shape and much less saddle points, according to [2]
    2) used batch normalization(BN, normalization over all training data in one mini-batch)[7] to make the algorithm converge fast, due to some possible reason:
        --by making the shape of objective function more symmetric which means where ever from to start searching, the converge speed should be similar rather\
        than somewhere converge too slow.
        --by making the activation to be followed by same 'mean' and 'variance' in one mini-batch, thus allocate larger scale for the gradient of activation which\
        should be smaller, to prevent vanishing of gradient problem
        --if we do NOT use batch normalization on each of layer, then the bias and variance of input of each layer will be highly dependent on previous layer's\
        weights/parameters, this could introduce very complex nonlinearity, which makes the network hard to train and might overfitting. By introducing BN, the bias\
        and varaince of input of each layer will be indepedent on previous layer's weights/parameters, so de-composite the complexity of network thus easy to train\
        and slightly avoid overfitting


The purpose is just showing the performance of ResNet on a simple dataset, which is so called CIFAR-10.
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training 
images and 10000 test images. 10 classes are 'airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
Each of picture is 32x32 pixel with 3 channels(R,G,B). CIFAR-10 could be downloaded in: http://www.cs.toronto.edu/~kriz/cifar.html
"""
"-------------------------------------------------------------------------------------------------"
"""Reference: 
    [1] 2015 Deep Residual Learning for Image Recognition. download here: https://arxiv.org/pdf/1512.03385.pdf"
        tutorial online: https://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf
    [2] 2017 Convergence Analysis of Two-layer Neural Networks with ReLU Activation. https://arxiv.org/pdf/1705.09886.pdf
        presentation: https://www.youtube.com/watch?v=pWxSUZWOctk
    [7] 2015 Batch normalization: Accelerating deep network training by reducing internal covariate shift
        https://arxiv.org/pdf/1502.03167.pdf
"""

"""
Note:
    a) A possible error about "Broken pips" and "multi-processing" can be happened, see this blog
    https://medium.com/@mackie__m/running-a-cifar-10-image-classifier-on-windows-with-pytorch-9094e29089cd and this
    https://discuss.pytorch.org/t/brokenpipeerror-errno-32-broken-pipe-when-i-run-cifar10-tutorial-py/6224 how to solve it
    b) Error 'Can't pickle <class>: it's not the same object, see this blog 
    https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
    how to solve it
    c) in a function that expects a list of items, how can I pass a Python list item iteratively without getting an error? e.g. 
    my_list = ['red', 'blue', 'orange']
    function_that_needs_strings('red', 'blue', 'orange') # works!
    function_that_needs_strings(my_list) # error!
    answer: function_that_needs_strings(*my_list) # works!
    see more information: https://stackoverflow.com/questions/3480184/unpack-a-list-in-python
    
"""

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import torchvision as tv
import torchvision.transforms as transforms
# from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

"-------------------------------------------------------------------------------------------------"
"""""""""""""""""""""""""""
1. Data preprocessing part
"""""""""""""""""""""""""""
pilTrans = transforms.ToPILImage()

"The output of torchvision datasets are PILImage images of range [0, 1]."
"We transform them to Tensors of normalized range [-1, 1]."
transform = transforms.Compose([
        transforms.ToTensor(), #-----conver PILImage to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #----- Normlization
                             ])

"torchvision package has dataset CIFAR10"

"make training set and load them by using DataLoader and defining the batch size, etc"
trainset = tv.datasets.CIFAR10(
                    root='./CIFAR10data', 
                    train=True, 
                    download=True,
                    transform=transform) #----- download=True could let script to download CIFAR-10 in the first time usage

trainloader = tc.utils.data.DataLoader(
                    trainset, 
                    batch_size=32,
                    shuffle=True, 
                    num_workers=0) #----- trying to iterate over the DataLoader with multiple num_workers, set as 0 could avoid some error when do multi-processing


"make dev/test set and load them by using DataLoader and defining the batch size, etc"
testset = tv.datasets.CIFAR10(
                    './CIFAR10data',
                    train=False, 
                    download=True, 
                    transform=transform) #-----download=True could let script to download CIFAR-10 in the first time usage

testloader = tc.utils.data.DataLoader(
                    testset,
                    batch_size=32, 
                    shuffle=False,
                    num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


(data, label) = trainset[100]
print(classes[label])

pil_data = pilTrans((data + 1) / 2).resize((100, 100))
# pilTrans(data)

plt.imshow(pil_data)



"""""""""""""""""""""""""""""""""""""""
2. Define ResNet34 architecture part
"""""""""""""""""""""""""""""""""""""""
"Note: all size of in and out channels, stride, padding, etc, could refer to table 1 from Kaiming[1], size of each layer output depends on input data"
"Residual block, as submodule"
class Residual_Block(nn.Module):
    def __init__(self, number_in_channel, number_out_channel, stride = 1, linear_projection = None): #----- __init__ consist of declaration of components in network
        super(Residual_Block, self).__init__()     #----- Call the constructor of base class explicitly
        self.normal_path = nn.Sequential(
                nn.Conv2d(number_in_channel, number_out_channel, 3, stride, 1, bias = False), #---- No need to make bias learnable, due using BatchNorm
                nn.BatchNorm2d(number_out_channel),
                nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                nn.Conv2d(number_out_channel, number_out_channel, 3, 1, 1, bias = False),
                nn.BatchNorm2d(number_out_channel) )
        self.linear_projection_on_shortcut_path = linear_projection
        
    def forward(self, x):
        out = self.normal_path(x)
        # residual = x if self.shortcut_path is None else self.shortcut_path(x)
        if self.linear_projection_on_shortcut_path == None:
            residual_link = x
        else:
            residual_link = self.linear_projection_on_shortcut_path(x)        
        out = out + residual_link
        return F.relu(out)
    
        
"ResNet34"
class ResNet34(nn.Module):                   #----- Define a Net class as derived class inherited from nn.Module
    "Note: The ResNet34 contains several parts in each part contains several residual blocks, see ResNet_Components.PNG"
    def __init__(self, num_classes = 10):                 #----- __init__ define the constructor of Net class, consist of declaration of components in network              
        super(ResNet34, self).__init__()     #----- Call the constructor of base class explicitly
        
        "declaration of the first non-residual normal block"
        self.normal_block = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False), #-----padding with 3 and stride 2 size shrink
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        
        "declaration of the rest parts which consist of residual blocks"
        self.part1 = self.make_residual_part(64, 64, 3)
        self.part2 = self.make_residual_part(64, 128, 4, stride = 2)
        self.part3 = self.make_residual_part(128, 256, 6, stride = 2)
        self.part4 = self.make_residual_part(256, 512, 3, stride = 2)
        
        "fully connected layers as classifier"
        self.fc = nn.Linear(in_features = 512, out_features = num_classes)

    def make_residual_part(self, number_in_channel, number_out_channel, number_of_residual_blocks, stride = 1, padding = 0):
        linear_projection = None        
        if (stride != 1) or (number_in_channel != number_out_channel): 
        #----- in case the stride is NOT 1, or number_in_channel is NOT same as number_out_channel, adapte the size and number of out channel of residual link
            linear_projection = nn.Sequential(
                    nn.Conv2d(number_in_channel, number_out_channel, 1, stride, padding = 0, bias = False),
                    nn.BatchNorm2d(number_out_channel))
            
        parts = []
        parts.append(Residual_Block(number_in_channel, number_out_channel, stride, linear_projection))
        for i in range(1, number_of_residual_blocks):
            parts.append(Residual_Block(number_out_channel, number_out_channel))
        return nn.Sequential(*parts) #----- iteratively pass each element in the list, see https://stackoverflow.com/questions/3480184/unpack-a-list-in-python for detail
        
    "All the forward propagation behavier is implemented in this function"
    def forward(self, x):
        # input(training/unkown data) --> conv1 --> ReLu --> Pooling --> conv2 --> ReLu --> Pooling --> fc1 --> fc2 --> fc3 --> lable y as output
        x = self.normal_block(x) #----- x --> normal_block --> result stores back in x
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self.part4(x)
        
        "Note: In Kaiming's paper there is last avg pool layer. However, ResNet34 doesn't need this avg pool layer in case using CIFAR10 data set" 
        "which consist of image in 32x32, because the output size from part4 is already 1x1 in 512 out channels"
        # x = F.avg_pool2d(x, kernel_size = 7) 
        
        x = x.view(x.size()[0], -1)                     #----- Reshape the x to make it flat
        x = self.fc(x)
        
        return x #----- return the x which is just right before passing the last activation function layer


     
our_resnet = ResNet34()
print('this is our ResNet: ', our_resnet)



"""""""""""""""""""""""""""""""""""""""
3. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
# optimizer = opt.SGD(our_lenet.parameters(), lr = 0.001, momentum=0.9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
optimizer = opt.Adam(our_resnet.parameters(), lr = 0.001, eps = 1e-08)    #----- use Adam algorithm for all parameters of our_classifier

"set a loss function"
# loss_function = nn.MSELoss()        #----- here use MSE loss
loss_function = nn.CrossEntropyLoss()        #----- here use cross entropy loss



"""""""""""""""""""""""""""
4. Train the ResNet34 part
"""""""""""""""""""""""""""
"Train the ResNet34"
tc.set_num_threads(10)  #----- Sets the number of OpenMP threads used for parallelizing CPU operations
for epoch in range(90):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        "load input data"
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        
        "clear all stored gradients if there exist"
        optimizer.zero_grad()
        
        "forward prop"
        outputs = our_resnet(inputs)
        
        "calculate the gradients for all Variables during back prop"
        loss = loss_function(outputs, labels)
        
        "back prop"
        loss.backward()
        
        "update all Variables by using newly fetched gradients"
        optimizer.step()
        
        "print log info"
        running_loss += loss.data[0]
        if i % 500 == 0: #----- print log info every 1000 batch
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / 500))
            running_loss = 0.0
            
print("training complete")



"Test with dev/test data"
correct_predication = 0 #
total_number_of_test_data = 0 
for data in testloader:
    images, labels = data
    outputs = our_resnet(Variable(images))
    _, predicted_y = tc.max(outputs.data, dim = 1)
    total_number_of_test_data += labels.size(0)
    correct_predication += (predicted_y  == labels).sum()

print('The accuracy of classification of dev/test data: %d %%' % (100 * correct_predication / total_number_of_test_data))



"accurate rate with train data"
correct_train = 0 #
total_number_of_train_data = 0 
for data in trainloader:
    images, labels = data
    outputs = our_resnet(Variable(images))
    _, predicted_y = tc.max(outputs.data, dim = 1)
    total_number_of_train_data += labels.size(0)
    correct_train += (predicted_y  == labels).sum()

print('The accuracy of classification of train data: %d %%' % (100 * correct_train / total_number_of_train_data))




