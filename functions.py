#import time
#import shutil
#from collections import OrderedDict, namedtuple
#import plotly
#import os
#import sys
#import os.path 
#import glob
#from shutil import copyfile
#from tqdm import tqdm
#import torchvision

import base64

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import transforms
from torchvision.models import vgg19

import pandas as pd
import numpy as np
import cv2

from PIL import Image
from skimage.transform import resize

from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import slic

import plotly.graph_objs as go
from plotly.offline import plot, iplot

import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst import utils
import segmentation_models_pytorch as smp


# pre and post processing for images
def prepare_image(img, img_size = (224, 224)):
    """
    Function, that transform image to tensor, that needed for 
    VGG19 network
    Parameters:
    -----------
    img: PIL.JpegImagePlugin.JpegImageFile
        Image.
    img_size: tuple
        Desired image size.
    Output: torch.Tensor
    """
    prep = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                 std=[1,1,1]),
            transforms.Lambda(lambda x: x.mul_(255)),
        ]
    )
    imgs_torch = prep(img)
    img_torch = Variable(imgs_torch.unsqueeze(0).cuda()) 
    return img_torch

def plot_distribution(img_path):
    """
    Function draw class distribution  for image
    Parameters:
    -----------
    img_path : str
        Path to input picture.

    """
    img = Image.open(img_path)
    img_width, img_height = img.size
    img = prepare_image(img = img)
    model = vgg19(pretrained=True).cuda().eval() 
    predict = model.forward(img)
    predict = predict.detach().cpu().numpy().reshape(-1)
    
    label = pd.read_csv('./label.csv', sep = ';', index_col=0)
    label['predict'] = predict
    label.sort_values(by = 'predict', inplace = True)
    trace = go.Bar(x = [str(i) + '_' + j for i, j in enumerate(label.label)], y = label.predict)
    l = go.Layout(
        title = 'Class distribution',
        xaxis = dict(
            title = 'Class'
        ),
        yaxis = dict(
            title = 'Score'
        )
    )
    fig = go.Figure(data = [trace], layout = l)
    iplot(fig)
    
def create_tensor(x, mean, v_shift, h_shift, grid_size, mask_height = 28, mask_width = 28):
    """
    Function create list of tensors with different region droped. 
    First element of that list is x.
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor.
    mean : bool
        How we should erase pixels? If False then erase by zero, else by mean.
    v_shift : int
        Vertical shift.
    h_shift : int
        Horizontal shift.
    grid_size : int
        Size of grid grid_size x grid_size.
    mask_height : int
        Mask height.
    mask_width : int
        Mask width.
    """
    size = mask_height // grid_size
    if not mean:
        mask = torch.ones((1,1,mask_height, mask_height)).cuda()
        result = [torch.clone(x) * mask]
        for i in range(grid_size + (v_shift % size  != 0)):
            for j in range(grid_size + (h_shift % size  != 0)):
                newmask = torch.clone(mask)
                newmask[0, 0, max(0, i*size - v_shift) : min((i+1)*size - v_shift, mask_height),
                              max(0, j*size - h_shift) : min((j+1)*size - h_shift, mask_width)] = 0
                result += [torch.clone(x) * newmask] 
    else:
        result = [torch.clone(x)]    
        for i in range(grid_size + (v_shift % size != 0)):
            for j in range(grid_size + (h_shift % size != 0)):
                new_x = torch.clone(x)
                mean = torch.mean(new_x[:, :, max(0, i*size - v_shift) : min((i+1)*size - v_shift, mask_height),
                                        max(0, j*size - h_shift) : min((j+1)*size - h_shift, mask_width)],
                                  dim = (0, 2, 3), keepdim = True)
                new_x[:, :, max(0, i*size - v_shift) : min((i+1)*size - v_shift, mask_height),
                            max(0, j*size - h_shift) : min((j+1)*size - h_shift, mask_width)] = mean
                result += [new_x]
    return result

def visual_importance(scale_factor, patch_size, img_path, name, mean, save_as_pic = False):
    """
    Function draw importance distribution on image.
    Parameters:
    -----------
    scale_factor : int
        How match compress image from zero to one.
    patch_size : int 
        Size of patch, should be dividers of 28.
    img_path : str
        Path to input picture.
    name : str
        Name of result picture, that would be saved in report directory.
    mean : bool
        How we should erase pixels? If False then erase by zero, else by mean.
    save_as_pic : bool
        Should we save image as picture or plot as figure.
        If True, than save as picture in report directory.
    """
    img = Image.open(img_path)
    img_width, img_height = img.size
    img = prepare_image(img)
    
    model = eval_importance().cuda()
    res = model.forward(img, 0, 0, 28//patch_size, mean)
    importance = np.array([float(torch.norm(res[0] - res[k + 1])) for k in range((28//patch_size)**2)])    

    shapes = []
    for i in range(28 // patch_size):
        for j in range(28 // patch_size):
            k = (28 // patch_size) * i + j
            shapes += [
                {
                    'type': 'rect',
                    'x0': j * img_width*scale_factor / (28 // patch_size),
                    'y0': ((28 // patch_size) - 1 - i) * img_height * scale_factor / (28 // patch_size),
                    'x1': (j + 1) * img_width*scale_factor / (28 // patch_size),
                    'y1': ((28 // patch_size) - i) * img_height * scale_factor / (28 // patch_size),
                    'line': {
                        'color': 'white' ,
                        'width': 2,
                    },
                    'fillcolor': 'rgba(255, 255, 255,' + str(0.8 * importance[k] / importance.max()) + ')'
                }
            ]
    
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    #add the prefix that plotly will want when using the string as source
    encoded_image = "data:image/png;base64," + encoded_string

    layout = go.Layout(
        shapes = shapes,
        xaxis = go.layout.XAxis(
            visible = False,
            range = [0, img_width*scale_factor]),
        yaxis = go.layout.YAxis(
            visible=False,
            range = [0, img_height*scale_factor],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor = 'x'),
        width = img_width*scale_factor,
        height = img_height*scale_factor,
        margin = {'l': 0, 'r': 0, 't': 0, 'b': 0},
        images = [dict(
            x=0,
            sizex=img_width*scale_factor,
            y=img_height*scale_factor,
            sizey=img_height*scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=encoded_image
        )]
    )

    # we add a scatter trace with data points in opposite corners to give the Autoscale feature a reference point
    fig = go.Figure(data=[{
        'x': [0, img_width * scale_factor], 
        'y': [0, img_height * scale_factor], 
        'mode': 'markers',
        'marker': {'opacity': 0}}
        ],layout = layout
        )
    if save_as_pic:
        fig.write_image(f'./report/pictures/{name}_{patch_size}.png')
    else:
        iplot(fig)

def compare_batch_size(img_path, name, mean = False, patch_sizes = [14, 7, 4, 2, 1]):
    """
    Function allows compare different patch sizes in terms of important
    
    Parameters:
    ----------
    img_path : str
        Path to input picture
    name : str
        Name of result picture, that would be saved in report directory
    mean : bool
        How we should erase pixels? If False then erase by zero, else by mean. 
    patch_sizes : list
        List of patch sizes. Every patch size should be dividers of 28.
    """
    for patch_size in patch_sizes:
         visual_importance(
             scale_factor = 1,
             patch_size = patch_size,
             img_path = img_path,
             name = name,
             mean = mean,
             save_as_pic = True
         )
    
    img = cv2.imread(img_path)
    
    for patch_size in patch_sizes:
        img = np.concatenate(
            (img, cv2.imread(f"./report/pictures/{name}_{patch_size}.png")),
            axis = 1
        )
    img = Image.fromarray(img[:, :, ::-1], 'RGB')
    img.save(f'./report/{name}.png')
    
# VGG network for content and style loss calculation 
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
    
# Gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

def weighted_mse(y, target, w):            
    """
    Weighted MSE
    """
    loss = w * (y - target) ** 2
    return loss.mean()

def compute_weight_simple(img, img_path, grid_size, img_width, img_height, mean):
    """
    Function calculate simple weight matrix for content loss
    Параметры:
    ----------
    img : torch.tensor
        Input image
    img_path : str
        Path to input picture
    grid_size : int
        Size of grid grid_size x grid_size
    img_width : int
        Width
    img_height :int
        Height 
    mean : bool
        How we should erase pixels? If False then erase by zero, else by mean
    """
    model = eval_importance().cuda()
    w = np.ones((1, 1, 28, 28), dtype = np.float32)

    res = model.forward(img, 0, 0, grid_size, mean)
    importance = np.array([float(torch.norm(res[0] - res[k + 1])) for k in range(len(res) - 1)])    
    device = torch.device('cuda')

    k = 0
    size = 28 // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            w[0, 0, i*size : (i+1)*size, j*size :(j+1)*size] = importance[k]
            k += 1
    w = resize(w[0,0,:,:], (img_height, img_width), preserve_range  = True)[np.newaxis, np.newaxis, :, :]

        
    w = w  / w.sum() * img_width * img_height
    w = torch.from_numpy(w).cuda().float()
    return w


def compute_weight(img, img_path, grid_size, contrast, img_width, img_height, mean):
    """
    Function calculate weight matrix for content loss using moving patch method
    Parameters:
    ----------
    img : torch.tensor
        Input image
    img_path : str
        Path to input picture
    grid_size : int
        Size of grid grid_size x grid_size
    contrast: int
       Contrast coefficient. Default value equal to one.
       Bigger value means, that front object will have bigger values in weight matrix.
    img_width : int
        Width
    img_height :int
        Height 
    mean : bool
        How we should erase pixels? If False then erase by zero, else by mean
    """
    model = eval_importance().cuda()
    v_size = img_height // grid_size
    h_size = img_width // grid_size
    size = 28 // grid_size
    w = np.ones((1, size ** 2, 28, 28), dtype = np.float32)
    layer = 0
    
    for v_shift in range(size):
        for h_shift in range(size):     
            res = model.forward(img, v_shift, h_shift, grid_size, mean)
            importance = np.array([float(torch.norm(res[0] - res[k + 1])) for k in range(len(res) - 1)])    
            k = 0
            
            for i in range(grid_size + (v_shift % size != 0)):
                for j in range(grid_size + (h_shift % size != 0)):
                    w[0, layer, max(0, i*size - v_shift) : min((i+1)*size - v_shift, img_height),
                                max(0, j*size - h_shift) : min((j+1)*size - h_shift, img_width)] = importance[k]
                    k += 1
            layer += 1
    w = np.mean(w, axis = 1)[:, np.newaxis, :, :]
    w = resize(w[0,0,:,:], (img_height, img_width), preserve_range  = True)[np.newaxis, np.newaxis, :, :]

    w = w  / w.sum() * img_width * img_height
    # Increase contrast
    wn = np.clip(contrast*(w - w.mean()) + w.mean(), 1e-1, w.sum())
#     wn = wn / wn.sum() * w.sum()
    w = torch.from_numpy(wn.copy()).cuda().float()
    return w

class eval_importance(torch.nn.Module):
    def __init__(self):
        super(eval_importance, self).__init__()
        model = vgg19(pretrained = True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        features = list(model.features)
        classifier = list(model.classifier)
        self.features = nn.ModuleList(features).eval() 
        self.classifier = nn.ModuleList(classifier).eval() 
    def forward(self, x, v_shift, h_shift, grid_size, mean):
        """
        Function takes image as input and return list of class probability distribution.
        First element of that list is input probability distribution.
        Other elemnts are probability distribution of erased part image in r42 layer
        Parameters:
        -----------
        x : torch.Tensor
            input image
        v_shift : int
            Vertical shift
        h_shift : int
            Horizontal shift
        grid_size : int
            Size of grid grid_size x grid_size
        mean : bool
            How we should erase pixels? If False then erase by zero, else by mean.
        """
        # Go to r42 layer
        for ii,model in enumerate(self.features[:23]):
            x = model(x)
        
        # Erase patches
        result = []
        tensor_list = create_tensor(x, mean, v_shift, h_shift, grid_size = grid_size)
        for new_x in tensor_list:
            # go to the end of convolutional nn   
            for ii,model in enumerate(self.features[23:]):
                new_x = model(new_x)

            new_x = self.avgpool(new_x)
            new_x = new_x.view(new_x.size(0), -1)
            # make classification
            for ii,model in enumerate(self.classifier):
                new_x = model(new_x)
            result += [new_x]

        return result
    
            
def create_superpixel_tensor(x, segments, mask_height = 224, mask_width = 224):
    """
    Function create list of tensors, that splited by superpixels.
    First tensor is just x.
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor
    segments : np.array
        List of superpixel segmentaion.
    mask_height : int
        Mask height
    mask_width : int
        Mask width
    """
    mask = torch.ones((1,1,mask_height, mask_width)).cuda()
    result = [torch.clone(x) * mask]
    for i in torch.unique(segments, sorted=True):
        newmask = torch.clone(mask)
        newmask[segments == i] = 0
        result += [torch.clone(x) * newmask] 
    return result

class eval_superpixel_importance(torch.nn.Module):
    def __init__(self):
        super(eval_superpixel_importance, self).__init__()
        model = vgg19(pretrained = True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        features = list(model.features)
        classifier = list(model.classifier)
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval() 
        self.classifier = nn.ModuleList(classifier).eval() 
    def forward(self, x, segments):
        """
        Function takes image as input and return list of class probability distribution.
        First element of that list is input probability distribution.
        Other elemnts are probability distribution of erased part image in r42 layer
        Параметры:
        ----------
        x : torch.Tensor
            Input tensor
        segments : np.array
            Superpixel split list
        """
        # Fist run
        result = []
        xx = torch.clone(x)
        for ii,model in enumerate(self.features):
            x = model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # Make classification
        for ii,model in enumerate(self.classifier):
            x = model(x)
        result += [x.cpu().detach().numpy()]
        # Second run
        for i in torch.unique(segments, sorted=True):
            mask = torch.ones((1,1,224, 224)).cuda()
            mask[segments == i] = 0
            x = xx * mask
            for ii,model in enumerate(self.features):
                x = model(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # Make classification
            for ii,model in enumerate(self.classifier):
                x = model(x)
            result += [x.cpu().detach().numpy()]
        del x, xx, mask, segments
        return result
    

def compute_superpixel_weight(img, img_path, img_width, img_height, contrast = 1):
    """
    Function calculate weight matrix for content loss using superpixel method
    Parameters:
    ----------
    img : torch.tensor
        Input image
    img_path : str
        Path to input picture
    img_width : int
        Width of image
    img_height :int   
        Height of image
    contrast: int
        Contrast coefficient. Default value equal to one.
        Bigger value means, that front object will have bigger values in weight matrix.
    """
    model = eval_superpixel_importance().cuda()
    
    W = np.zeros((img_height, img_width))
    image = img_as_float(io.imread(img_path))
    image = resize(image, (224, 224), preserve_range = True)
    k = 0
    for sigma in np.linspace(0, 5, 10):
        for n_segments in np.arange(10, 100, 10):
            segments = slic(image, n_segments = n_segments, sigma = sigma)[np.newaxis, np.newaxis, :, :]
            res = model.forward(img, torch.from_numpy(segments).cuda())
            importance = np.array([np.sqrt(np.sum((res[0] - res[k + 1])**2)) for k in range(len(res) - 1)])    
            w = np.copy(segments)
            for i, superpixel in enumerate(np.unique(segments)):
                w[w == superpixel] = importance[i]

            w = resize(w[0,0,:,:], (img_height, img_width), preserve_range  = True)

            w = w  / w.sum() * img_width * img_height
            W += w
            k += 1
    # Add contrast    
    w = W / k
    wn = np.clip(contrast*(w - w.mean()) + w.mean(), 1e-1, w.sum())[np.newaxis, np.newaxis, :, :]
    wn = torch.from_numpy(wn).cuda().float()
    
    del model, W, w
    return wn
    
def compute_segmenation_weight(img_path, img_width, img_height, contrast = 1):
    """
    Function calculate weight matrix for content loss using fully 
    convolutional neural network
    Parameters:
    ----------
    img_path : str
        Path to input picture
    img_width : int
        Width of image
    img_height :int   
        Height of image

    contrast: int
        Contrast coefficient. Default value equal to one.
        Bigger value means, that front object will have bigger values in weight matrix.
    """
    default_image_size = 320
    def pre_transforms(image_size=default_image_size):
        return [albu.Resize(image_size, image_size, p=1)]

    def post_transforms():
        # we use ImageNet image normalization
        # and convert it to torch.Tensor    
        return [albu.Normalize(), ToTensor()]
      
    def compose(transforms_to_compose):
        # combine all augmentations into one single pipeline
        result = albu.Compose([
          item for sublist in transforms_to_compose for item in sublist
        ])
        return result
        
    valid_transforms = compose([pre_transforms(), post_transforms()])
    model = smp.FPN(encoder_name="resnext101_32x8d", classes=1)
    model.load_state_dict(torch.load(f'./segmentation/model.pth'))
    model.eval()
    
    image = {'image':utils.imread(img_path)}
    x = valid_transforms(**image)['image']
    y = model.predict(x.view(1, 3, 320, 320))[0, 0].sigmoid().numpy()
    w = resize(y, (img_height, img_width), preserve_range  = True)
    w = w  / w.sum() * img_width * img_height
    wn = np.clip(contrast*(w - w.mean()) + w.mean(), 1e-1, w.sum())[np.newaxis, np.newaxis, :, :]
    wn = torch.from_numpy(wn).cuda().float()
    
    del model, x, y, w
    return wn

def postp(tensor): 
    """
    to clip results in the range [0,1]
    """
    postpa = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
    ])
    postpb = transforms.Compose([transforms.ToPILImage()])
    
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img
 
 
def stylization(img_path, style_img_path, directory, name, weight_function, contrast = 1, img_size = 640, grid_size = 4):
    """
    Function, that make stylization of pic_path with style_pic_path.
    Parameters:
    -----------
    img_path : str
        Path to content image
    style_img_path : str
        Path to style image
    directory : str
        Directory where result image would be saved
    name : str
        Name of result image
    weight_function : str
        Type of function, that would be used for finding weight matrix
        Possible values:
            - 'patch' : simple patch method with grid_size
            - 'moving_patch' : moving patch method with grid_size
            - 'superpixel' : superpixel method
            - 'segmentation' : segmentation method
            - 'gatys' : standard Gatys algorithm

    contrast : float
        Contrast coefficient. Default value equal to one.
        Bigger value means, that front object will have bigger values in weight matrix.
    img_size : int    
        Minimal size side of resulting picture
    grid_size : int
        Size of grid grid_size x grid_size. If batch size equal to zero,
        than classic Gatys stylization method is applied 
    """
    def closure(w):
        optimizer.zero_grad()
        out = vgg(opt_img, style_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        out = vgg(opt_img, content_layers)
        layer_losses += [weighted_mse(out[0], content_targets[0], w)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        return loss
    
    max_iter = 500
        
    prep = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
        transforms.Lambda(lambda x: x.mul_(255)),
    ])    
    
    #load images, ordered as [style_image, content_image]
    img_names = [style_img_path, img_path]
    imgs = [Image.open(name) for name in img_names]
    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch
    opt_img = Variable(content_image.data.clone(), requires_grad=True)
    #vgg network for loss fucntion calculation
    vgg = VGG()
    vgg.load_state_dict(torch.load('./Models/vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()
    
    #define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11','r21','r31','r41', 'r51'] 
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [weighted_mse] 
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns[:-1]] + [loss_fns[-1]]

    #these are good weights settings:
    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    #compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets
    
    img = Image.open(img_path)
    img_width, img_height = img.size
    img = prepare_image(img = img)
    
    if weight_function == 'patch':
        w = compute_weight_simple(
            img, img_path, grid_size, 
            content_targets[0].shape[3], content_targets[0].shape[2], False
        )
    elif weight_function == 'moving_patch':
        w = compute_weight(
            img, img_path, grid_size, contrast,
            content_targets[0].shape[3], content_targets[0].shape[2], False
        )
    elif weight_function == 'superpixel':
        grid_size = ''
        w = compute_superpixel_weight(
            img, img_path, content_targets[0].shape[3],
            content_targets[0].shape[2], contrast = contrast
        )
    elif weight_function == 'segmentation':
        grid_size = ''
        w = compute_segmenation_weight(
            img_path, content_targets[0].shape[3],
            content_targets[0].shape[2], contrast
        )
    elif weight_function == 'gatys':
        grid_size = ''
        w = torch.ones(1, 1, content_targets[0].shape[2], content_targets[0].shape[3], dtype = torch.float32).cuda()
    else:
        print(f'weight_function = {weight_function} is wrong value')
        return -1

    optimizer = optim.LBFGS([opt_img])
    n_iter=[0]
    while n_iter[0] <= max_iter:
        optimizer.step(lambda: closure(w))

    out_img = postp(opt_img.data[0].cpu().squeeze())
    out_img.save(f'{directory}{weight_function}_{name}_{grid_size}.png')

    del vgg, w, out_img, img, targets, content_targets, style_targets, weights, content_weights, style_weights