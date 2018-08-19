import json
import re
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
import torchvision

def save_weights(model, model_path, ep, step, train_metrics, valid_metrics):
    torch.save({'model': model.state_dict(), 'epoch': ep, 'step': step, 'valid_loss': valid_metrics['loss1'], 'train_loss': train_metrics['loss1']},
               str(model_path)
               )


def write_tensorboard(writer, model, epoch, train_metrics, valid_metrics):
    ## epoch starts from 1, not 0
    ## save the results on epoch 1
    if (epoch-1) % 5 == 0:
        train_image, train_mask, train_prob = train_metrics['image'], train_metrics['mask'], train_metrics['prob']
        valid_image, valid_mask, valid_prob = valid_metrics['image'], valid_metrics['mask'], valid_metrics['prob']
        saved_images = torchvision.utils.make_grid(train_image, nrow=train_image.shape[0], padding=10, pad_value=1)
        writer.add_image('train/Image', saved_images, epoch)
        ######
        for n in range(train_mask.shape[1]):
            saved_images = torch.cat((train_mask.narrow(1,n,1), train_prob.narrow(1, n, 1)), 0)
            saved_images = torchvision.utils.make_grid(saved_images, nrow=train_image.shape[0], padding=10, pad_value=1)
            writer.add_image('train/Mask%s' % n, saved_images, epoch)
        ##### test
        saved_images = torchvision.utils.make_grid(valid_image, nrow=valid_image.shape[0], padding=10, pad_value=1)
        writer.add_image('test/Image', saved_images, epoch)
        ######
        ######
        for n in range(valid_mask.shape[1]):
            saved_images = torch.cat((valid_mask.narrow(1, n, 1), valid_prob.narrow(1, n, 1)), 0)
            saved_images = torchvision.utils.make_grid(saved_images, nrow=valid_image.shape[0], padding=10, pad_value=1)
            writer.add_image('test/Mask%s' % n, saved_images, epoch)
        ######
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    if (epoch-1) % 1 == 0:
        #####
        valid_loss, valid_loss1, valid_loss2, valid_loss3 = valid_metrics['loss'],valid_metrics['loss1'],valid_metrics['loss2'],valid_metrics['loss3']
        train_loss, train_loss1, train_loss2, train_loss3 = train_metrics['loss'],train_metrics['loss1'],train_metrics['loss2'],train_metrics['loss3']
        train_jaccard, valid_jaccard = train_metrics['jaccard'],valid_metrics['jaccard']
        writer.add_scalars('loss', {'train': train_loss, 'test': valid_loss}, epoch)
        writer.add_scalars('loss1',{'train': train_loss1, 'test': valid_loss1},epoch)
        writer.add_scalars('loss2', {'train': train_loss2, 'test': valid_loss2}, epoch)
        writer.add_scalars('loss3', {'train': train_loss3, 'test': valid_loss3}, epoch)

        for out in range(1,3):
            for auc in range(1,6):
                key = 'out{}auc{}'.format(str(out), str(auc))
                writer.add_scalars(key, {'train': train_metrics[key],'test': valid_metrics[key]}, epoch)

        writer.add_scalars('jaccard', {'train': train_jaccard,'test': valid_jaccard}, epoch)
        #####


def write_event(log, step, epoch, train_metrics, valid_metrics):
    # train_metrics['step'] = step
    # train_metrics['epoch'] = epoch
    # train_metrics['valid_metrics'] = step
    # train_metrics['valid_metrics'] = epoch
    # train_metrics['dt'] = datetime.now().isoformat()
    # valid_metrics['dt'] = datetime.now().isoformat()
    # log.write(json.dumps(train_metrics, sort_keys=True))
    # log.write(json.dumps(valid_metrics, sort_keys=True))
    # log.write('\n')
    # log.flush()
    #print(data['loss'])
    CMD = 'epoch:{} step:{} time:{:.2f} train_loss:{:.3f} {:.3f} {:.3f} {:.3f} train_auc1:{} {} {} {} {} train_auc2:{} {} {} {} {} train_jaccard:{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n valid_loss:{:.3f} {:.3f} {:.3f} {:.3f} valid_auc1:{} {} {} {} {} valid_auc2:{} {} {} {} {} valid_jaccard:{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
        epoch, step, train_metrics['epoch_time'],
        train_metrics['loss'],train_metrics['loss1'],train_metrics['loss2'],train_metrics['loss3'],
        train_metrics['out1auc1'],train_metrics['out1auc2'],train_metrics['out1auc3'],train_metrics['out1auc4'],train_metrics['out1auc5'],
        train_metrics['out2auc1'],train_metrics['out2auc2'],train_metrics['out2auc3'],train_metrics['out2auc4'],train_metrics['out2auc5'],
        train_metrics['jaccard'],train_metrics['jaccard1'],train_metrics['jaccard2'],train_metrics['jaccard3'],train_metrics['jaccard4'],train_metrics['jaccard5'],
        valid_metrics['loss'], valid_metrics['loss1'], valid_metrics['loss2'], valid_metrics['loss3'],
        valid_metrics['out1auc1'], valid_metrics['out1auc2'], valid_metrics['out1auc3'], valid_metrics['out1auc4'],
        valid_metrics['out1auc5'],
        valid_metrics['out2auc1'], valid_metrics['out2auc2'], valid_metrics['out2auc3'], valid_metrics['out2auc4'],
        valid_metrics['out2auc5'],
        valid_metrics['jaccard'],valid_metrics['jaccard1'],valid_metrics['jaccard2'],valid_metrics['jaccard3'],valid_metrics['jaccard4'],valid_metrics['jaccard5'],
    )
    print(CMD)
    log.write(json.dumps(CMD))
    # log.write(json.dumps(valid_metrics, sort_keys=True))
    log.write('\n')
    log.flush()
    #for keys in sorted(train_metrics):
     #   print('{}:{}'.format(keys,train_metrics[keys]))

def print_model_summay(model):
    for name, para in model.named_parameters():
        print(name, para.numel(), para.requires_grad)
    return None

def set_freeze_layers(model, freeze_layer_names=None):
    for name, para in model.named_parameters():
        #print(name, para.numel(), para.requires_grad)
        if freeze_layer_names is None:
            para.requires_grad = True
        else:
            if name in freeze_layer_names:
                print('Freeze layer->', name)
                para.requires_grad = False
            else:
                para.requires_grad = True
    ## no need to return the model
    ## model has been modified within the function
    ## pass by reference
    #return model

def set_train_layers(model, train_layer_names=None):
    for name, para in model.named_parameters():
        #print(name, para.numel(), para.requires_grad)
        if train_layer_names is None:
            para.requires_grad = False
        else:
            if name in train_layer_names:
                print('Train layer ->', name)
                para.requires_grad = True
            else:
                para.requires_grad = False
    ## no need to return the model
    ## model has been modified within the function
    ## pass by reference
    #return model


def get_freeze_layer_names(model,part):

    #
    # if part == 'encoder':
    #     layers = {
    #     'module.encoder.0.weight':False,
    #     'module.encoder.0.bias':False,
    #     'module.encoder.2.weight':False,
    #     'module.encoder.2.bias':False,
    #     'module.encoder.5.weight':False,
    #     'module.encoder.5.bias':False,
    #     'module.encoder.7.weight':False,
    #     'module.encoder.7.bias':False,
    #     'module.encoder.10.weight':False,
    #     'module.encoder.10.bias':False,
    #     'module.encoder.12.weight':False,
    #     'module.encoder.12.bias':False,
    #     'module.encoder.14.weight':False,
    #     'module.encoder.14.bias':False,
    #     'module.encoder.17.weight':False,
    #     'module.encoder.17.bias':False,
    #     'module.encoder.19.weight':False,
    #     'module.encoder.19.bias':False,
    #     'module.encoder.21.weight':False,
    #     'module.encoder.21.bias':False,
    #     'module.encoder.24.weight':False,
    #     'module.encoder.24.bias':False,
    #     'module.encoder.26.weight':False,
    #     'module.encoder.26.bias':False,
    #     'module.encoder.28.weight':False,
    #     'module.encoder.28.bias':False,
    #     'module.center.block.0.conv.weight':True,
    #     'module.center.block.0.conv.bias':True,
    #     'module.center.block.1.weight':True,
    #     'module.center.block.1.bias':True,
    #     'module.center_Conv2d.weight':True,
    #     'module.center_Conv2d.bias':True,
    #     'module.dec5.block.0.conv.weight':True,
    #     'module.dec5.block.0.conv.bias':True,
    #     'module.dec5.block.1.weight':True,
    #     'module.dec5.block.1.bias':True,
    #     'module.dec4.block.0.conv.weight':True,
    #     'module.dec4.block.0.conv.bias':True,
    #     'module.dec4.block.1.weight':True,
    #     'module.dec4.block.1.bias':True,
    #     'module.dec3.block.0.conv.weight':True,
    #     'module.dec3.block.0.conv.bias':True,
    #     'module.dec3.block.1.weight':True,
    #     'module.dec3.block.1.bias':True,
    #     'module.dec2.block.0.conv.weight':True,
    #     'module.dec2.block.0.conv.bias':True,
    #     'module.dec2.block.1.weight':True,
    #     'module.dec2.block.1.bias':True,
    #     'module.dec1.conv.weight':True,
    #     'module.dec1.conv.bias':True,
    #     'module.final.weight':True,
    #     'module.final.bias':True
    #     }
    # if part == 'decoder':
    #     layers = {
    #         'module.encoder.0.weight': True,
    #         'module.encoder.0.bias': True,
    #         'module.encoder.2.weight': True,
    #         'module.encoder.2.bias': True,
    #         'module.encoder.5.weight': True,
    #         'module.encoder.5.bias': True,
    #         'module.encoder.7.weight': True,
    #         'module.encoder.7.bias': True,
    #         'module.encoder.10.weight': True,
    #         'module.encoder.10.bias': True,
    #         'module.encoder.12.weight': True,
    #         'module.encoder.12.bias': True,
    #         'module.encoder.14.weight': True,
    #         'module.encoder.14.bias': True,
    #         'module.encoder.17.weight': True,
    #         'module.encoder.17.bias': True,
    #         'module.encoder.19.weight': True,
    #         'module.encoder.19.bias': True,
    #         'module.encoder.21.weight': True,
    #         'module.encoder.21.bias': True,
    #         'module.encoder.24.weight': True,
    #         'module.encoder.24.bias': True,
    #         'module.encoder.26.weight': True,
    #         'module.encoder.26.bias': True,
    #         'module.encoder.28.weight': True,
    #         'module.encoder.28.bias': True,
    #         'module.center.block.0.conv.weight': True,
    #         'module.center.block.0.conv.bias': True,
    #         'module.center.block.1.weight': True,
    #         'module.center.block.1.bias': True,
    #         'module.center_Conv2d.weight': True,
    #         'module.center_Conv2d.bias': True,
    #         'module.dec5.block.0.conv.weight': False,
    #         'module.dec5.block.0.conv.bias': False,
    #         'module.dec5.block.1.weight': False,
    #         'module.dec5.block.1.bias': False,
    #         'module.dec4.block.0.conv.weight': False,
    #         'module.dec4.block.0.conv.bias': False,
    #         'module.dec4.block.1.weight': False,
    #         'module.dec4.block.1.bias': False,
    #         'module.dec3.block.0.conv.weight': False,
    #         'module.dec3.block.0.conv.bias': False,
    #         'module.dec3.block.1.weight': False,
    #         'module.dec3.block.1.bias': False,
    #         'module.dec2.block.0.conv.weight': False,
    #         'module.dec2.block.0.conv.bias': False,
    #         'module.dec2.block.1.weight': False,
    #         'module.dec2.block.1.bias': False,
    #         'module.dec1.conv.weight': False,
    #         'module.dec1.conv.bias': False,
    #         'module.final.weight': False,
    #         'module.final.bias': False
    #     }
    freeze_layers = []
    for ind, (name, para) in enumerate(model.named_parameters()):
        if re.search(part, name):
            #print(ind, name, para.numel(), para.requires_grad)
            freeze_layers.append(name)
        if part == 'encoder':
            if re.search('center', name):
                freeze_layers.append(name)
    # for ly in layers:
    #     if not layers[ly]:
    #         freeze_layers.append(ly)
    return freeze_layers