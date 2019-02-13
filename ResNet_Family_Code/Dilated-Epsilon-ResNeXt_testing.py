# -*- coding: utf-8 -*-
"-------------------------------------------------------------------------------------------------"
"""
Dilated-Epsilon-ResNeXt Classifier
"""
"""
Author: jianan.liu
Version: 1.0.0
"""
"-------------------------------------------------------------------------------------------------"
"""
This is a demo code of Dilated-Epsilon-ResNeXt.
    1) Dilated-Epsilon-ResNeXt model is combination between Dilated ResNet, ResNeXt and Epsilon-ResNet model proposed in [13,14], [8] and [11].
       Note: Same as Dilated-ResNet original paper[14], we also use Dilated conv to replace only the max pooling layers in the last 2 parts(groups) of original Epsilon-ResNeXt as well.
       Note: We don't replace the first max pooling layer of normal_block(the max pool layer follower 7x7 conv), so it is basically Dilated-ResNet a] structure rather than b] c][14]
       Note: In case of normal residual block, both part3 and part4 have 2 normal residual blocks. The first layer in the first normal residual block under part3 and part4 has 1 dilation if dilation rate is 2.
             has 2 dilation if dilation rate is 4.
    2) Dilated-ResNet[14] and Dilated-CNN[13] are proposed in 2017, 2015. Dilated convolution actually implements convolution with holes, is used to replace the downsampling(pooling layer).
       By oding this, Dilated-ResNet increases the resolution of output feature maps(cause prevents the elimination of spatial acuity, so it has higher chance to achieve better accuracy) 
       without reducing the receptive field of individual neurons(just use  stride = 1 for pooling layer/"removing subsampling" could also maintain higher resolution of feature maps, however
       this reduces the receptive field in subsequent layers). In other words, Dilated conv could support the exponetially increasing of receptive field, but unlike pooling, 
       Dilated conv not loss any resolution 
    3) ResNeXt[8] in theory, works better than ResNet[1](which works as almost convex problem[2]), almost equals to performance of DenseNet[10] as comparaion of 6.figure of[9].
    4) Epsilon-ResNet[11] is proposed for reducing the size/number of parameter used in the network architecture. The "gate" residual block in Epsilon-ResNet could be implemented by
       using multiplication between sparse promoting fucntion (which consists of 4 ReLU) and the input into sparse promoting function itself(which is output from previous layer), 
       as written in [11] and [12]. 
       Note This is like pruning technique in machine learning algorithm desicen tree.
    5) used batch normalization(BN, normalization over all training data in one mini-batch)[7] to make the algorithm converge fast, due to some possible reason:
        --by making the shape of objective function more symmetric which means where ever from to start searching, the converge speed should be similar rather\
        than somewhere converge too slow.
        --by making the activation to be followed by same 'mean' and 'variance' in one mini-batch, thus allocate larger scale for the gradient of activation which\
        should be smaller, to prevent vanishing of gradient problem
        --if we do NOT use batch normalization on each of layer, then the bias and variance of input of each layer will be highly dependent on previous layer's\
        weights/parameters, this could introduce very complex nonlinearity, which makes the network hard to train and might overfitting. By introducing BN, the bias\
        and varaince of input of each layer will be indepedent on previous layer's weights/parameters, so de-composite the complexity of network thus easy to train\
        and slightly avoid overfitting
    6) In this version, the "cardinality" is implemented by using property "group" in PyTorch for each conv layer
    
    @TODO:
        1) consider Kaiming's initialization[3] to 
           how to use initilization for Pytorch? https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        2) consider dropout layer to prevent overfitting[6]
        4) consider added L2 regularization to avoid overfitting,
        5) consider to switch from Adam in the early training iterations to SGD in the late training iterations, inspired from [4].
           By doing this, it may enhance the generilization ability of network on performance in test/dev data
        6) consider making ResNet50, by adding “bottleneck” building block reducing the consumption of computation by using
           multiple smalll size conv filter to replace one big size conv filter[1]
        7) might consider just using residual link in the shallow layers rather than using residual link in both shallow layers(e.g. first 30% layers) 
           and deep layers(e.g. the deeper 70% layers), the reason is only residual link in shallow layers are actually passing the value actively 
           according to [5]

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
    [3] 2015 Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. https://arxiv.org/pdf/1502.01852.pdf
    [4] 2017 Improving Generalization Performance by Switching from Adam to SGD. https://arxiv.org/pdf/1712.07628.pdf
    [5] 2016 Residual Networks Behave Like Ensembles of Relatively Shallow Networks. 
        https://papers.nips.cc/paper/6556-residual-networks-behave-like-ensembles-of-relatively-shallow-networks.pdf
        https://zhuanlan.zhihu.com/p/37820282
    [6] 2014 Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    [7] 2015 Batch normalization: Accelerating deep network training by reducing internal covariate shift
        https://arxiv.org/pdf/1502.03167.pdf
    [8] 2016 Aggregated Residual Transformations for Deep Neural Networks
        https://arxiv.org/pdf/1611.05431.pdf
    [9] Memory-Efficient Implementation of DenseNets
        https://arxiv.org/pdf/1707.06990.pdf
    [10]2016 Densely Connected Convolutional Networks. https://arxiv.org/pdf/1608.06993.pdf
        source code: https://github.com/chisyliu/DenseNet
    [11]2018 Learning Strict Identity Mappings in Deep Residual Networks. https://arxiv.org/abs/1804.01661
    [12]2015 Highway Networks. https://arxiv.org/abs/1505.00387
    [13]2017 Dilated Residual Networks. https://arxiv.org/abs/1705.09914
    [14]2015 Multi-Scale Context Aggregation by Dilated Convolutions. https://arxiv.org/abs/1511.07122
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
2. Define ResNeXt architecture part
"""""""""""""""""""""""""""""""""""""""
"Note: all size of in and out channels, stride, padding, etc, could refer to table 1 from Kaiming[1], size of each layer output depends on input data"
"Residual block, as submodule"
class Normal_Residual_Block(nn.Module): #----- In this version, the "cardinality" is implemented by using property "group" in PyTorch for each conv layer
    def __init__(self, number_in_channel, number_out_channel, stride = 1, linear_projection = None, num_of_group = 32, gate_in_use = True, EPSILON = 1, dilation_rate = 1, dilation_rate_second_layer = 1):
        #----- num_of_group is "cardinality"
        #----- dilation_rate_second_layer is the second layer of each residual block's dilation rate
        super(Normal_Residual_Block, self).__init__()     #----- Call the constructor of base class explicitly
        self.normal_path = nn.Sequential(
                nn.Conv2d(number_in_channel, number_out_channel, 3, stride, padding = dilation_rate, bias = False, dilation = dilation_rate), #---- No need to make bias learnable, due using BatchNorm
                nn.BatchNorm2d(number_out_channel),
                nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                nn.Conv2d(number_out_channel, number_out_channel, 3, 1, padding = dilation_rate_second_layer, bias = False, groups = num_of_group, dilation = dilation_rate_second_layer),
                nn.BatchNorm2d(number_out_channel) )
        self.linear_projection_on_shortcut_path = linear_projection
        self.gate_in_use = gate_in_use
        self.epsilon = EPSILON
        
    def forward(self, x):
        out = self.normal_path(x)
        # residual = x if self.shortcut_path is None else self.shortcut_path(x)
        if self.linear_projection_on_shortcut_path == None:
            residual_link = x
        else:
            residual_link = self.linear_projection_on_shortcut_path(x)

        "Gate on or off"
        if self.gate_in_use == True:
            y = tc.max(tc.nn.functional.relu(out - self.epsilon, inplace = True) + tc.nn.functional.relu(-out - self.epsilon, inplace = True))
            gate_on_off_result = tc.nn.functional.relu(tc.nn.functional.relu(y * (-10000000) + 1) * (-10000000) + 1)
            out = gate_on_off_result * out + residual_link
        else:
            out = out + residual_link
        return F.relu(out)


class BottleNeck_Residual_Block(nn.Module): #----- In this version, the "cardinality" is implemented by using property "group" in PyTorch for each conv layer
    "Note: number_out_channel must be even number"
    def __init__(self, number_in_channel, number_out_channel, stride = 1, linear_projection = None, num_of_group = 32, gate_in_use = True, EPSILON = 1, dilation_rate = 1, dilation_rate_second_layer = 1): 
        #----- num_of_group is "cardinality"
        #----- dilation_rate_second_layer is the second layer of each residual block's dilation rate. Note it is useless for bottleNeck residual block cause "only the middle layer in bottleneck is useful layer with dilation rate"
        super(BottleNeck_Residual_Block, self).__init__()     #----- Call the constructor of base class explicitly
        if number_out_channel % 2 == 1:
            print('Warning: number_out_channel for Bottle Neck Residual Block is NOT even number')
        self.normal_path = nn.Sequential(
                nn.Conv2d(number_in_channel, int(number_out_channel/2), 1, 1, 0, bias = False), #---- No need to make bias learnable, due using BatchNorm
                nn.BatchNorm2d(int(number_out_channel/2)),
                nn.ReLU(inplace = True), #-----inplace = True could overwrite the input of ReLU by using output to save memory(ReLU only needs output to calculate gradient )
                nn.Conv2d(int(number_out_channel/2), int(number_out_channel/2), 3, stride, padding = dilation_rate, groups = num_of_group, bias = False, dilation = dilation_rate),
                nn.BatchNorm2d(int(number_out_channel/2)),
                nn.ReLU(inplace = True),
                nn.Conv2d(int(number_out_channel/2), number_out_channel, 1, 1, 0, bias = False),
                nn.BatchNorm2d(number_out_channel))
        self.linear_projection_on_shortcut_path = linear_projection
        self.gate_in_use = gate_in_use
        self.epsilon = EPSILON
        
    def forward(self, x):
        out = self.normal_path(x)
        #-- residual = x if self.shortcut_path is None else self.shortcut_path(x)
        if self.linear_projection_on_shortcut_path == None:
            residual_link = x
        else:
            residual_link = self.linear_projection_on_shortcut_path(x)
            
        "Gate on or off"
        if self.gate_in_use == True:
            y = tc.max(tc.nn.functional.relu((out - self.epsilon), inplace = True) + tc.nn.functional.relu(((-1) * out - self.epsilon), inplace = True))
            print(y)
            gate_on_off_result = tc.nn.functional.relu(tc.nn.functional.relu(y * (-10000000) + 1) * (-10000000) + 1)
            out = gate_on_off_result * out + residual_link
        else:
            out = out + residual_link
        return F.relu(out)
    
        
"ResNeXt34"
class ResNeXt34(nn.Module):                   #----- Define a Net class as derived class inherited from nn.Module
    "Residual_Block_Type is either class 'BottleNeck_Residual_Block' or 'class Normal_Residual_Block"
    def __init__(self, Residual_Block_Type, num_classes = 10):                 #----- __init__ define the constructor of Net class, consist of declaration of components in network              
        super(ResNeXt34, self).__init__()     #----- Call the constructor of base class explicitly
        
        "declaration of the first non-residual normal block"
        self.normal_block = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False), #-----size shrink 1/2
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        
        "declaration of the rest parts which consist of residual blocks"
        self.part1 = self.make_residual_part(Residual_Block_Type, 64, 64, 3, 32, gate_in_use = False)
        self.part2 = self.make_residual_part(Residual_Block_Type, 64, 128, 4, 32, stride = 2)
        self.part3 = self.make_residual_part(Residual_Block_Type, 128, 256, 6, 32, stride = 1, dilation_rate = 2)
        self.part4 = self.make_residual_part(Residual_Block_Type, 256, 512, 3, 32, stride = 1, gate_in_use = True, dilation_rate = 4)
        last_layer_dilation_rate = 4
        
        "fully connected layers as classifier"
        self.fc = nn.Linear(in_features = 512*last_layer_dilation_rate*last_layer_dilation_rate, out_features = num_classes) #----- the in_features = 512*last_layer_dilation_rate*last_layer_dilation_rate

    def make_residual_part(self, Residual_Block_Type, number_in_channel, number_out_channel, number_of_residual_blocks, num_of_group, stride = 1, padding = 0, gate_in_use = True, EPSILON = 1, dilation_rate = 1):
        linear_projection = None        
        if (stride != 1) or (number_in_channel != number_out_channel): 
        #----- in case the stride is NOT 1, or number_in_channel is NOT same as number_out_channel, adapte the size and number of out channel of residual link
            linear_projection = nn.Sequential(
                    nn.Conv2d(number_in_channel, number_out_channel, 1, stride, padding = 0, bias = False),
                    nn.BatchNorm2d(number_out_channel))
            
        parts = []
        
        if dilation_rate == 1 or dilation_rate == 2:
            parts.append(Residual_Block_Type(number_in_channel, number_out_channel, stride, linear_projection, num_of_group, gate_in_use, EPSILON, dilation_rate = 1, dilation_rate_second_layer = dilation_rate))
        elif dilation_rate == 4:
            parts.append(Residual_Block_Type(number_in_channel, number_out_channel, stride, linear_projection, num_of_group, gate_in_use, EPSILON, dilation_rate = 2, dilation_rate_second_layer = dilation_rate))
        else:
            print('Warning: Not supportive dilation rate')
            
        for i in range(1, number_of_residual_blocks):
            parts.append(Residual_Block_Type(number_out_channel, number_out_channel, num_of_group = num_of_group, gate_in_use = gate_in_use, EPSILON = EPSILON, dilation_rate = dilation_rate, dilation_rate_second_layer = dilation_rate))
        return nn.Sequential(*parts) #----- iteratively pass each element in the list, see https://stackoverflow.com/questions/3480184/unpack-a-list-in-python for detail
        
    "All the forward propagation behavier is implemented in this function"
    def forward(self, x):
        # input(training/unkown data) --> conv1 --> ReLu --> Pooling --> conv2 --> ReLu --> Pooling --> fc1 --> fc2 --> fc3 --> lable y as output
        x = self.normal_block(x) #----- x --> normal_block --> result stores back in x
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self.part4(x)
        
        "Note: In Kaiming's paper there is last avg pool layer. However, ResNeXt34 doesn't need this avg pool layer in case using CIFAR10 data set" 
        "which consist of image in 32x32, because the output size from part4 is already 1x1 in 512 out channels"
        # x = F.avg_pool2d(x, kernel_size = 7) 
        
        x = x.view(x.size()[0], -1)                     #----- Reshape the x to make it flat
        x = self.fc(x)
        
        return x #----- return the x which is just right before passing the last activation function layer


     
"Residual_Block_Type is either class 'BottleNeck_Residual_Block' or 'class Normal_Residual_Block'"
our_resnext = ResNeXt34(BottleNeck_Residual_Block)
# our_resnext = ResNeXt34(Normal_Residual_Block)
print('this is our e-ResNeXt: ', our_resnext)



"""""""""""""""""""""""""""""""""""""""
3. Setup optimization algorithm part
"""""""""""""""""""""""""""""""""""""""
"set an optimizer"
optimizer = opt.SGD(our_resnext.parameters(), lr = 0.0001, momentum=0.9)    #----- use SGD algorithm for all parameters of our_lenet, by learning rate 0.01 and Momentum is 0.9
# optimizer = opt.Adam(our_resnext.parameters(), lr = 0.001, eps = 1e-08)    #----- use Adam algorithm for all parameters of our_classifier

"set a loss function"
# loss_function = nn.MSELoss()        #----- here use MSE loss
loss_function = nn.CrossEntropyLoss()        #----- here use cross entropy loss



"""""""""""""""""""""""""""
4. Train the ResNeXt34 part
"""""""""""""""""""""""""""
"Train the ResNeXt34"
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
        outputs = our_resnext(inputs)
        
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
    outputs = our_resnext(Variable(images))
    _, predicted_y = tc.max(outputs.data, dim = 1)
    total_number_of_test_data += labels.size(0)
    correct_predication += (predicted_y  == labels).sum()

print('The accuracy of classification of dev/test data: %d %%' % (100 * correct_predication / total_number_of_test_data))



"accurate rate with train data"
correct_train = 0 #
total_number_of_train_data = 0 
for data in trainloader:
    images, labels = data
    outputs = our_resnext(Variable(images))
    _, predicted_y = tc.max(outputs.data, dim = 1)
    total_number_of_train_data += labels.size(0)
    correct_train += (predicted_y  == labels).sum()

print('The accuracy of classification of train data: %d %%' % (100 * correct_train / total_number_of_train_data))




