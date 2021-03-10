import os
import math
import csv
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
# from graphviz import Digraph

import logging
from collections import OrderedDict
from PIL import Image
import numpy as np
import cv2

from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx
import re

def visualize_gnn(h, color, epoch=None, loss=None, with_labels=False):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=True,
                         node_color=color, cmap="Set2")
    plt.show()

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
        img = np.concatenate([img, img, img], axis=-1)

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    
    return cam

def merge(image1, image2):
    if len(image1.shape) == 3:
        if image1.shape[-1] == 4:
            image1 = image1[:, :, :3]
    else:
        image1 = np.stack([image1, image1, image1], axis=-1)

    if len(image2.shape) == 3:
        if image2.shape[-1] == 4:
            image2 = image2[:, :, :3]  
    else:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    image1 = normalize(image1)
    image2 = normalize(image2) 

    image_merge = np.concatenate((image1, image2), axis=1) 
    
    return image_merge


def draw_two_contours(image, mask, gt, color=(0, 0, 255), thickness=1):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.uint8(image * 255)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    im = np.copy(image)
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = np.uint8(mask)
    if len(gt.shape) == 3:
        gt = gt[:, :, 0]
    gt = np.uint8(gt)

    contours_mask, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(im, contours_mask, -1, (255, 0, 0), thickness=thickness)
    image_with_contours = cv2.drawContours(image_with_contours, contours_gt, -1, color, thickness=thickness)

    return image_with_contours

def draw_contours(image, mask, color=(255, 255, 0), thickness=1):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.uint8(image * 255)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    im = np.copy(image)
    
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = np.uint8(mask)

    contours_mask, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(im, contours_mask, -1, (255, 0, 0), thickness=thickness)

    return image_with_contours


def adjust_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def label_smooth(label, limit = 0.2):
    device = label.device
    random_noise = torch.rand(label.size()).to(device) * limit
    label = torch.abs(label - random_noise)
    
    return label

class MultiTargetCrossEntropyLoss():
    def __init__(self):
        self.lsoftmax = nn.LogSoftmax(dim=-1)
    
    def __call__(self, input, target):
        log_pred = self.lsoftmax(input)
        loss = torch.sum(((-log_pred) * target)) / len(target)

        return loss

class CSVLogger(object):
    def __init__(self, log_path, header):
        self.log_file = open(log_path, mode = 'w')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(header)

    def write_row(self, msg):
        self.writer.writerow(msg)
        self.log_file.flush()

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x

def one_hot_classification(label, num_cls):
    batch_size = label.size(0)
    label = label.long().view(-1, 1)
    out_tensor = torch.zeros(batch_size, num_cls).to(label.device)
    out_tensor.scatter_(1, label, 1)

    return out_tensor

def one_hot_segmentation(label, num_cls):
    batch_size = label.size(0)
    label = label.long()
    out_tensor = torch.zeros(batch_size, num_cls, *label.size()[2:]).to(label.device)
    out_tensor.scatter_(1, label, 1)

    return out_tensor

def find_lr(model, optimizer, criterion, train_loader, device, epochs = 1, init_value = 1e-8, final_val = 10., beta = 0.98):
    #plot learning rate versus loss
    def plt_lr_curve(lr_record, losses):
        plt.semilogx(lr_record, losses)
        plt.show()

    # start
    num = (len(train_loader) - 1) * epochs
    mult = (final_val / init_value) ** (1/num)
    lr = init_value
    for params in optimizer.param_groups:
        params['lr'] = lr
    avg_loss = 0.
    best_loss = math.inf
    losses_record = []
    lrs_record = []
    
    model.train()
    
    for epoch in range(epochs):

        batch_num = 0

        for data in train_loader:
            batch_num += 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # highlight this 
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (len(train_loader) * epoch + batch_num))
            # Stop if the loss is exploding
            if smoothed_loss > 4 * best_loss:
                # plot learning rate versus loss
                plt_lr_curve(lrs_record, losses_record)

                return lrs_record, losses_record
            # Update best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            # Append records
            losses_record.append(smoothed_loss)
            lrs_record.append(lr)
            # Update model params
            loss.backward()
            optimizer.step()
            # Update learning rate
            lr *= mult
            for params in optimizer.param_groups:
                params['lr'] = lr

    # plot learning rate versus loss
    plt_lr_curve(lrs_record, losses_record)

    return lrs_record, losses_record

def rampup_function(epochs, curr_epoch, max_rampup_val = 8.0):
    if curr_epoch < epochs:
        p = max(0.0, float(curr_epoch)) / float(epochs)
        p = 1.0 - p
        rampup_factor = math.exp(-p*p*5.0)
    else:
        rampup_factor = 1.0

    return max_rampup_val * rampup_factor

def visualize_dataloader(dataloader, example_num, figsize=(10, 10), normalize=False):
    data = next(iter(dataloader))
    imgs = data[0][:example_num]
    labels = data[-1][:example_num]

    imgs = vutils.make_grid(imgs, normalize=normalize)
    imgs = np.transpose(imgs, (1, 2, 0))
    masks = vutils.make_grid(labels, normalize=normalize)
    masks = np.transpose(masks, (1, 2, 0))

    plt.figure(figsize = figsize)
    plt.subplot(211)
    plt.imshow(imgs, cmap='gray')
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(masks, cmap='gray')
    plt.axis('off')

    plt.show()

    return np.unique(imgs), np.unique(masks)

def save_tensor_to_image(tensor1, tensor2, image_dir, image_index, example_num=4):
    N, C, D, H, W = tensor1.size()
    tensor1 = torch.transpose(tensor1, 1, 2)
    tensor2 = torch.transpose(tensor2, 1, 2)
    tensor1 = tensor1.view(-1, C, H, W)
    tensor2 = tensor2.view(-1, C, H, W)
    case = torch.randint(low=0, high=D, size=(example_num,))

    img1 = torch.stack([normalize(tensor1[i]) for i in case])
    img2 = torch.stack([normalize(tensor2[i]) for i in case])

    img1 = vutils.make_grid(img1, normalize=False)
    img2 = vutils.make_grid(img2, normalize=False)
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))

    image_dir = os.path.join(image_dir, 'images')
    create_dir([image_dir])
    img1_path = os.path.join(image_dir, f"img1_{image_index}.png")
    img2_path = os.path.join(image_dir, f"img2_{image_index}.png")

    plt.imsave(img1_path, img1.numpy())
    plt.imsave(img2_path, img2.numpy())

def visualize_3d_tensor(tensor1, tensor2, example_num=4, figsize=(8, 8)):
    N, C, D, H, W = tensor1.size()
    tensor1 = torch.transpose(tensor1, 1, 2)
    tensor2 = torch.transpose(tensor2, 1, 2)
    tensor1 = tensor1.view(-1, C, H, W)
    tensor2 = tensor2.view(-1, C, H, W)
    case = torch.randint(low=0, high=D, size=(example_num,))

    img1 = torch.stack([normalize(tensor1[i]) for i in case])
    img2 = torch.stack([normalize(tensor2[i]) for i in case])

    img1 = vutils.make_grid(img1, normalize=False)
    img2 = vutils.make_grid(img2, normalize=False)
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))

    plt.figure(figsize = figsize)
    plt.subplot(211)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')

    plt.show()

def visualize_single_tensor(tensor, example_num=4, figsize=(8, 8), normalize=False):
    img = tensor[:example_num]
    
    img = vutils.make_grid(img, normalize=normalize)
    img = np.transpose(img, (1, 2, 0))

    plt.figure(figsize = figsize)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.show()

def save_paired_tensors(image_tensor, mask_tensor, filename):
    n = image_tensor.size(0)
    image_tensor = torch.gt(image_tensor, 0.5).int()
    mask_tensor = torch.gt(mask_tensor, 0.5).int()

    tensor = torch.cat([image_tensor, mask_tensor], dim=0)
    tensor = vutils.make_grid(tensor, nrow=n)
    tensor = np.transpose(tensor, (1, 2, 0))
    image = np.uint8(np.array(tensor) * 255)
    cv2.imwrite(filename, image)

def save_single_tensor(tensor, filename):
    tensor = torch.gt(tensor, 0.5).float()
    image = torch.squeeze(tensor).numpy()
    assert len(image.shape) == 2
    image = np.uint8(image * 255)
    flag = cv2.imwrite(filename, image)

def log_parse(log_dir, metric, best_type='min'):
    log_path = os.path.join(log_dir, 'log', 'train', 'Train.log')
    p = r'%s-(.{6})' % metric
    logs_list = []
    vals_list = []
    with open(log_path, 'r') as f:
        logs = f.readlines()
        for log in logs:
            res = re.search(p, log)
            if res:
                logs_list.append(log)
                vals_list.append(float(res.group(1)))

    if best_type == 'min':
        index = np.argsort(vals_list)[0]
    else:
        index = np.argsort(vals_list)[-1]
    
    return logs_list[index]

def make_dot(var, params=None):
  """ Produces Graphviz representation of PyTorch autograd graph
  Blue nodes are the Variables that require grad, orange are Tensors
  saved for backward in torch.autograd.Function
  Args:
    var: output Variable
    params: dict of (name, Variable) to add names to node that
      require grad (TODO: make optional)
  """
  if params is not None:
    assert isinstance(params.values()[0], Variable)
    param_map = {id(v): k for k, v in params.items()}
  
  node_attr = dict(style='filled',
           shape='box',
           align='left',
           fontsize='12',
           ranksep='0.1',
           height='0.2')
  dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
  seen = set()
  
  def size_to_str(size):
    return '('+(', ').join(['%d' % v for v in size])+')'
  
  def add_nodes(var):
    if var not in seen:
      if torch.is_tensor(var):
        dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
      elif hasattr(var, 'variable'):
        u = var.variable
        name = param_map[id(u)] if params is not None else ''
        node_name = '%s\n %s' % (name, size_to_str(u.size()))
        dot.node(str(id(var)), node_name, fillcolor='lightblue')
      else:
        dot.node(str(id(var)), str(type(var).__name__))
      seen.add(var)
      if hasattr(var, 'next_functions'):
        for u in var.next_functions:
          if u[0] is not None:
            dot.edge(str(id(u[0])), str(id(var)))
            add_nodes(u[0])
      if hasattr(var, 'saved_tensors'):
        for t in var.saved_tensors:
          dot.edge(str(id(t)), str(id(var)))
          add_nodes(t)
  add_nodes(var.grad_fn)

  return dot

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
