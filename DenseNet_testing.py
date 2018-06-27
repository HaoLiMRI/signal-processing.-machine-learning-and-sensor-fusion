# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
DenseNet-BC Classifier
"""
"""
Author: jianan.liu
Version: 1.0.0
"""
"-------------------------------------------------------------------------------------------------"
"""
This is a demo code of DenseNet-BC with L = 100, k = 12. where L is number of layers, k is number of feature maps(channels)/growth rate for output from bottlenect layer
    1) DenseNet[1] in theory, is logical extension of ResNet, works even better than ResNet, especially on the following items:
        a) Much less number of paramters(saving the memory and computation) needed compared with ResNet(just around half)
        b) The author of DenseNet found out, even randomly "drop/skip" some layers in ResNet(which leads to [4]), the ResNet still converge well\
           It means:
               ----When drop layer l, the output from layer l-1 will be connected to layer l+1 directly, and the network still works. So it means\
                   extraction of features is NOT dependent on the exact last layer but is dependent on even earlier layer. So why not just use all\
                   previous layer together to the latest layer to extract features? This is how the 'Dense connection' concept comes out
               ----the ResNet still converge well when randomly drop lots layers, so this means there are lots of redundancy in the ResNet\
                   the every layer in ResNet only extracts a little features/info, so why not just makes the channels of each layer's output\
                   smaller? This is how the small 'growth rate'(e.g. keep the channel of output of layer small) comes in DenseNet. In fact, \
                   Due to lots of connections provided from previous layers to each late layer, it is reasonable to just have small number of\
                   channels of output of every layer(this means each layer only needs to learn small amount of features). 
                   If there are not lots of 'dense connection', there is NO point to have small number of channels of output of every layer
               ----the DenseNet is good at generalization(avoid overfitting), especially when the data set is NOT big enough(which should easily overfitting)\.
                   General neural network classifier that directly depends on the feature extracted from last layer (which has the highest nonlinearity/complexity)
                   of the network. On contrast, DenseNet also scombines the features with low nonlinearity/complexity from the shallow layer, makes it easier to 
                   obtain a smooth hypothesis function with better generalization performance. 
    2) DenseNet-B stands for deisgning of DenseNet with inception mechanism(inception was proposed[6], the explaination could be found in[7]), that means 
        one dense block has 4 bottleneck layers each consist of 'BN ----> ReLU ----> 1X1 conv ----> BN ----> ReLU ----> 3x3 conv'
    3) DenseNet-C means designing of DenseNet with feature maps(channels) downsizing in transition layer, use '(optional)BN ----> 1x1 conv ----> 2x2 average pooling' 
        as transition layers between two contiguous dense blocks, and make transition layer generate only half of the feature maps generated from previous dense block
    3) For systematic evaluation of difference between VGG, ResNet, and DenseNet, etc, could be seen[5]

The purpose is just showing the performance of ResNet on a simple dataset, which is so called CIFAR-10.
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training 
images and 10000 test images. 10 classes are 'airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
Each of picture is 32x32 pixel with 3 channels(R,G,B). CIFAR-10 could be downloaded in: http://www.cs.toronto.edu/~kriz/cifar.html
"""
"-------------------------------------------------------------------------------------------------"
"""Reference: 
    [1] 2016 Densely Connected Convolutional Networks. https://arxiv.org/pdf/1608.06993.pdf
        source code: https://github.com/chisyliu/DenseNet
    [4] 2016 Deep Networks with Stochastic Depth. https://arxiv.org/pdf/1603.09382.pdf
    [5] 2018 A Systematic Evaluation of Recent Deep Learning Architectures for Fine-Grained Vehicle Classification
    [6] 2014 Going deeper with convolutions
    [7] Deeplearning.ai course, part 4.2
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
import math

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

"torchvision package has dataset CIFAR10, which has 50000 training/10000 test image in size 32 x 32 x 3 in 10 class"

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
                    transform=transform) #-----download = True could let script to download CIFAR-10 in the first time usage

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
2. Define DenseNet-BC architecture part
"""""""""""""""""""""""""""""""""""""""
"Note: all size of in and out channels, stride, padding, etc, could refer to , size of each layer output depends on input data"
"Residual block, as submodule"  
class Transition_Layer(nn.Module):
    def __init__(self, number_in_channel, theta = 0.5, stride = 1): #----- __init__ consist of declaration of components in network    
        super(Transition_Layer, self).__init__()
        self.bn = nn.BatchNorm2d(number_in_channel)
        self.conv = nn.Conv2d(number_in_channel, int(math.floor(theta*number_in_channel)), 1, stride, 0, bias = False)
        self.avg_pooling = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0) #---- size shrik as 1/2 of input
        
    
    "Not sure whether there is a ReLU between BN and conv in transition layer" 
    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.avg_pooling(x)
        return x
    

class Bottleneck_Layer(nn.Module):
    def __init__(self, number_in_channel, number_out_feature_map, stride_for_bottle_neck_path = 1): #----- __init__ consist of declaration of components in network
        super(Bottleneck_Layer, self).__init__()
        self.bottle_neck_path = nn.Sequential(
                nn.BatchNorm2d(number_in_channel),
                nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                nn.Conv2d(number_in_channel, 4*number_out_feature_map, 1, stride_for_bottle_neck_path, 0, bias = False)) #---- same padding
        
        self.normal_path = nn.Sequential(
                nn.BatchNorm2d(4*number_out_feature_map),
                nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                nn.Conv2d(4*number_out_feature_map, number_out_feature_map, 3, 1, 1, bias = False)) #---- same padding
        
    def forward(self, x):
        out = self.bottle_neck_path(x)
        out = self.normal_path(out)
        out = tc.cat((x, out), 1)
        return out

    
        
"DenseNet"
class DenseNet(nn.Module):                   #----- Define a Net class as derived class inherited from nn.Module
    "Note: The DenseNet-BC in detail could be seen in DenseNet_detail_config.PNG"
    def __init__(self, growth_rate = 12, num_classes = 10):                 #----- __init__ define the constructor of Net class, consist of declaration of components in network              
        super(DenseNet, self).__init__()     #----- Call the constructor of base class explicitly
        
        "declaration of the first seperate conv layer"
        number_out_channel_for_first_conv = 2*growth_rate
        self.part1 = nn.Conv2d(3, number_out_channel_for_first_conv, kernel_size = 3, stride = 1, padding = 1, bias = False) #---- 'same' padding, size does NOT shrik
        
        "declaration of the rest parts which consist of 3 dense blocks and 2 transition layers for DenseNet-BC, L = 100, k = 12"
        number_of_bottleneck_layers = 4
        theta = 0.5
        self.part2 = self.make_dense_block(number_out_channel_for_first_conv, growth_rate, number_of_bottleneck_layers) #----channel number of output from part2(first dense block) is (number_out_channel_for_first_conv + number_of_bottleneck_layers*growth_rate)
        
        number_out_chanel_for_first_dense_block = number_out_channel_for_first_conv + number_of_bottleneck_layers*growth_rate
        self.part3 = Transition_Layer(number_out_chanel_for_first_dense_block, theta, stride = 1) #---- channel number of output from part3(first transition layer) is int(math.floor(theta*(number_out_channel_for_first_conv + number_of_bottleneck_layers*growth_rate)))
        
        number_out_chanel_for_first_transition_layer = int(math.floor(theta*number_out_chanel_for_first_dense_block))
        self.part4 = self.make_dense_block(number_out_chanel_for_first_transition_layer, growth_rate, number_of_bottleneck_layers)
        
        number_out_chanel_for_second_dense_block = number_out_chanel_for_first_transition_layer + number_of_bottleneck_layers*growth_rate
        self.part5 = Transition_Layer(number_out_chanel_for_second_dense_block, theta, stride = 1)
        
        number_out_chanel_for_second_transition_layer = int(math.floor(theta*number_out_chanel_for_second_dense_block))
        self.part6 = self.make_dense_block(number_out_chanel_for_second_transition_layer, growth_rate, number_of_bottleneck_layers)
        
        "declaration of fully connected layers as classifier"
        number_out_chanel_for_third_dense_block = number_out_chanel_for_second_transition_layer + number_of_bottleneck_layers*growth_rate
        self.fc = nn.Linear(in_features = number_out_chanel_for_third_dense_block, out_features = num_classes)

    "In the paper, figure 1 figure 2 says one dense block consist of 4 dense layer"
    def make_dense_block(self, number_in_channel, growth_rate, number_of_bottleneck_layers = 4, stride_for_bottle_neck_path = 1):
        parts = []
        parts.append(Bottleneck_Layer(number_in_channel, growth_rate, stride_for_bottle_neck_path))
        for i in range(1, number_of_bottleneck_layers):
            parts.append(Bottleneck_Layer(growth_rate*i + number_in_channel, growth_rate, stride_for_bottle_neck_path))
        return nn.Sequential(*parts) #----- iteratively pass each element in the list, see https://stackoverflow.com/questions/3480184/unpack-a-list-in-python for detail
                                                        
    "All the forward propagation behavier is implemented in this function. Here the data is passing through the whole network\
    which consists of dense blocks and transition layers"
    def forward(self, x):
        "input(training/unkown data) --> part1(conv) --> part2(dense_block) --> part3(transition_layer) --> part4(dense_block) --> part5(transition_layer) -->\
        part6(dense_block) --> pooling --> fc --> lable y as output"
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self.part4(x)
        x = self.part5(x)
        x = self.part6(x)
        
        x = F.avg_pool2d(x, kernel_size = 8, stride = 8) #---- size in 8x8 shrik into 1x1         
        x = x.view(x.size()[0], -1)                     #----- Reshape the x to make it flat
        x = self.fc(x)
        
        return x #----- return the x which is just right before passing the last activation function layer


     
our_denseNet = DenseNet()
print('this is our DenseNet: ', our_denseNet)



"""""""""""""""""""""""""""""""""""""""
3. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
# optimizer = opt.SGD(our_lenet.parameters(), lr = 0.001, momentum=0.9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
optimizer = opt.Adam(our_denseNet.parameters(), lr = 0.001, eps = 1e-08)    #----- use Adam algorithm for all parameters of our_classifier

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
        outputs = our_denseNet(inputs)
        
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
    outputs = our_denseNet(Variable(images))
    _, predicted_y = tc.max(outputs.data, dim = 1)
    total_number_of_test_data += labels.size(0)
    correct_predication += (predicted_y  == labels).sum()

print('The accuracy of classification of dev/test data: %d %%' % (100 * correct_predication / total_number_of_test_data))



"accurate rate with train data"
correct_train = 0 #
total_number_of_train_data = 0 
for data in trainloader:
    images, labels = data
    outputs = our_denseNet(Variable(images))
    _, predicted_y = tc.max(outputs.data, dim = 1)
    total_number_of_train_data += labels.size(0)
    correct_train += (predicted_y  == labels).sum()

print('The accuracy of classification of train data: %d %%' % (100 * correct_train / total_number_of_train_data))




