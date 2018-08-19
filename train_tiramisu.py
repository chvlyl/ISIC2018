import argparse
import json
import random
from pathlib import Path
import numpy as np
import time

## pytorch
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.backends import cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter


## model
from models import UNet, UNet11, UNet16, LinkNet34
from loss import LossBinary
from dataset import make_loader
from utils import save_weights, write_event, write_tensorboard,print_model_summay,set_freeze_layers,set_train_layers,get_freeze_layer_names
from validation import validation_binary
from prepare_train_val import get_split
from transforms import DualCompose,ImageOnly,Normalize,HorizontalFlip,VerticalFlip
from metrics import AllInOneMeter


from model_tiramisu import FCDenseNet57, FCDenseNet67, FCDenseNet103


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', type=float, default=1)
    arg('--root', type=str, default='runs/debug', help='checkpoint root')
    arg('--image-path', type=str, default='data', help='image path')
    arg('--batch-size', type=int, default=2)
    arg('--n-epochs', type=int, default=100)
    arg('--optimizer', type=str, default='Adam', help='Adam or SGD')
    arg('--lr', type=float, default=0.001)
    arg('--workers', type=int, default=10)
    arg('--model', type=str, default='UNet16', choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34','FCDenseNet57','FCDenseNet67','FCDenseNet103'])
    arg('--model-weight', type=str, default=None)
    arg('--resume-path', type=str, default=None)
    arg('--attribute', type=str, default='all', choices=['pigment_network', 'negative_network',
                                                              'streaks', 'milia_like_cyst',
                                                              'globules', 'all'])
    args = parser.parse_args()


    ## folder for checkpoint
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    image_path = args.image_path

    #print(args)
    if args.attribute == 'all':
        num_classes = 5
    else:
        num_classes = 1
    args.num_classes = num_classes
    ### save initial parameters
    print('--' * 10)
    print(args)
    print('--' * 10)
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    ## load pretrained model
    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes, pretrained=True)
    elif args.model == 'FCDenseNet103':
        model = FCDenseNet103(num_classes=num_classes)
    else:
        model = UNet(num_classes=num_classes, input_channels=3)

    ## multiple GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    ## load pretrained model
    if args.model_weight is not None:
        state = torch.load(args.model_weight)
        #epoch = state['epoch']
        #step = state['step']
        model.load_state_dict(state['model'])
        print('--' * 10)
        print('Load pretrained model', args.model_weight)
        #print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        print('--' * 10)
        ## replace the last layer
        ## although the model and pre-trained weight have differernt size (the last layer is different)
        ## pytorch can still load the weight
        ## I found that the weight for one layer just duplicated for all layers
        ## therefore, the following code is not necessary
        # if args.attribute == 'all':
        #     model = list(model.children())[0]
        #     num_filters = 32
        #     model.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        #     print('--' * 10)
        #     print('Load pretrained model and replace the last layer', args.model_weight, num_classes)
        #     print('--' * 10)
        #     if torch.cuda.device_count() > 1:
        #         model = nn.DataParallel(model)
        #     model.to(device)

    ## model summary
    print_model_summay(model)

    ## define loss
    loss_fn = LossBinary(jaccard_weight=args.jaccard_weight)


    ## It enables benchmark mode in cudnn.
    ## benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the
    ## optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
    ## But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears,
    ## possibly leading to worse runtime performances.
    cudnn.benchmark = True

    ## get train_test_id
    train_test_id = get_split()

    ## train vs. val
    print('--' * 10)
    print('num train = {}, num_val = {}'.format((train_test_id['Split'] == 'train').sum(),
                                                (train_test_id['Split'] != 'train').sum()
                                                ))
    print('--' * 10)


    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        ImageOnly(Normalize())
    ])

    ## define data loader
    train_loader = make_loader(train_test_id, image_path, args, train=True, shuffle=True,
                               transform=train_transform)
    valid_loader = make_loader(train_test_id, image_path, args, train=False, shuffle=True,
                               transform=val_transform)

    if True:
        print('--'*10)
        print('check data')
        train_image, train_mask, train_mask_ind = next(iter(train_loader))
        print('train_image.shape', train_image.shape)
        print('train_mask.shape', train_mask.shape)
        print('train_mask_ind.shape', train_mask_ind.shape)
        print('train_image.min', train_image.min().item())
        print('train_image.max', train_image.max().item())
        print('train_mask.min', train_mask.min().item())
        print('train_mask.max', train_mask.max().item())
        print('train_mask_ind.min', train_mask_ind.min().item())
        print('train_mask_ind.max', train_mask_ind.max().item())
        print('--' * 10)

    valid_fn = validation_binary


    ###########
    ## optimizer
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

    ## loss
    criterion = loss_fn
    ## change LR
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)


    ##########
    ## load previous model status
    previous_valid_loss = 10
    model_path = root / 'model.pt'
    if args.resume_path is not None and model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        epoch = 1
        step = 0
        try:
            previous_valid_loss = state['valid_loss']
        except:
            previous_valid_loss = 10
        print('--' * 10)
        print('Restored previous model, epoch {}, step {:,}'.format(epoch, step))
        print('--' * 10)
    else:
        epoch = 1
        step = 0

    #########
    ## start training
    log = root.joinpath('train.log').open('at', encoding='utf8')
    writer = SummaryWriter()
    meter = AllInOneMeter()
    #if previous_valid_loss = 10000
    print('Start training')
    print_model_summay(model)
    previous_valid_jaccard = 0
    for epoch in range(epoch, args.n_epochs + 1):
        model.train()
        random.seed()
        #jaccard = []
        start_time = time.time()
        meter.reset()
        w1 = 1.0
        w2 = 0.5
        w3 = 0.5
        try:
            train_loss = 0
            valid_loss = 0
            # if epoch == 1:
            #     freeze_layer_names = get_freeze_layer_names(part='encoder')
            #     set_freeze_layers(model, freeze_layer_names=freeze_layer_names)
            #     #set_train_layers(model, train_layer_names=['module.final.weight','module.final.bias'])
            #     print_model_summay(model)
            # elif epoch == 5:
            #     w1 = 1.0
            #     w2 = 0.0
            #     w3 = 0.5
            #     freeze_layer_names = get_freeze_layer_names(part='encoder')
            #     set_freeze_layers(model, freeze_layer_names=freeze_layer_names)
            #     # set_train_layers(model, train_layer_names=['module.final.weight','module.final.bias'])
            #     print_model_summay(model)
            #elif epoch == 3:
            #     set_train_layers(model, train_layer_names=['module.dec5.block.0.conv.weight','module.dec5.block.0.conv.bias',
            #                                                'module.dec5.block.1.weight','module.dec5.block.1.bias',
            #                                                'module.dec4.block.0.conv.weight','module.dec4.block.0.conv.bias',
            #                                                'module.dec4.block.1.weight','module.dec4.block.1.bias',
            #                                                'module.dec3.block.0.conv.weight','module.dec3.block.0.conv.bias',
            #                                                'module.dec3.block.1.weight','module.dec3.block.1.bias',
            #                                                'module.dec2.block.0.conv.weight','module.dec2.block.0.conv.bias',
            #                                                'module.dec2.block.1.weight','module.dec2.block.1.bias',
            #                                                'module.dec1.conv.weight','module.dec1.conv.bias',
            #                                                'module.final.weight','module.final.bias'])
            #     print_model_summa zvgf    t5y(model)
            # elif epoch == 50:
            #     set_freeze_layers(model, freeze_layer_names=None)
            #     print_model_summay(model)
            for i, (train_image, train_mask, train_mask_ind) in enumerate(train_loader):
                # inputs, targets = variable(inputs), variable(targets)

                train_image = train_image.permute(0, 3, 1, 2)
                train_mask = train_mask.permute(0, 3, 1, 2)
                train_image = train_image.to(device)
                train_mask = train_mask.to(device).type(torch.cuda.FloatTensor)
                train_mask_ind = train_mask_ind.to(device).type(torch.cuda.FloatTensor)
                # if args.problem_type == 'binary':
                #     train_mask = train_mask.to(device).type(torch.cuda.FloatTensor)
                # else:
                #     #train_mask = train_mask.to(device).type(torch.cuda.LongTensor)
                #     train_mask = train_mask.to(device).type(torch.cuda.FloatTensor)

                outputs, outputs_mask_ind1, outputs_mask_ind2 = model(train_image)
                #print(outputs.size())
                #print(outputs_mask_ind1.size())
                #print(outputs_mask_ind2.size())
                ### note that the last layer in the model is defined differently
                # if args.problem_type == 'binary':
                #     train_prob = F.sigmoid(outputs)
                #     loss = criterion(outputs, train_mask)
                # else:
                #     #train_prob = outputs
                #     train_prob = F.sigmoid(outputs)
                #     loss = torch.tensor(0).type(train_mask.type())
                #     for feat_inx in range(train_mask.shape[1]):
                #         loss += criterion(outputs, train_mask)
                train_prob = F.sigmoid(outputs)
                train_mask_ind_prob1 = F.sigmoid(outputs_mask_ind1)
                train_mask_ind_prob2 = F.sigmoid(outputs_mask_ind2)
                loss1 = criterion(outputs, train_mask)
                #loss1 = F.binary_cross_entropy_with_logits(outputs, train_mask)
                #loss2 = nn.BCEWithLogitsLoss()(outputs_mask_ind1, train_mask_ind)
                #print(train_mask_ind.size())
                #weight = torch.ones_like(train_mask_ind)
                #weight[:, 0] = weight[:, 0] * 1
                #weight[:, 1] = weight[:, 1] * 14
                #weight[:, 2] = weight[:, 2] * 14
                #weight[:, 3] = weight[:, 3] * 4
                #weight[:, 4] = weight[:, 4] * 4
                #weight = weight * train_mask_ind + 1
                #weight = weight.to(device).type(torch.cuda.FloatTensor)
                loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, train_mask_ind)
                loss3 = F.binary_cross_entropy_with_logits(outputs_mask_ind2, train_mask_ind)
                #loss3 = criterion(outputs_mask_ind2, train_mask_ind)
                loss = loss1*w1 + loss2*w2 + loss3*w3
                #print(loss1.item(), loss2.item(), loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                #jaccard += [get_jaccard(train_mask, (train_prob > 0).float()).item()]
                meter.add(train_prob, train_mask, train_mask_ind_prob1, train_mask_ind_prob2, train_mask_ind,
                          loss1.item(),loss2.item(),loss3.item(),loss.item())
                # print(train_mask.data.shape)
                # print(train_mask.data.sum(dim=-2).shape)
                # print(train_mask.data.sum(dim=-2).sum(dim=-1).shape)
                # print(train_mask.data.sum(dim=-2).sum(dim=-1).sum(dim=0).shape)
                # intersection = train_mask.data.sum(dim=-2).sum(dim=-1)
                # print(intersection.shape)
                # print(intersection.dtype)
                # print(train_mask.data.shape[0])
                #torch.zeros([2, 4], dtype=torch.float32)
            #########################
            ## at the end of each epoch, evualte the metrics
            epoch_time = time.time() - start_time
            train_metrics = meter.value()
            train_metrics['epoch_time'] = epoch_time
            train_metrics['image'] = train_image.data
            train_metrics['mask'] = train_mask.data
            train_metrics['prob'] = train_prob.data

            #train_jaccard = np.mean(jaccard)
            #train_auc = str(round(mtr1.value()[0],2))+' '+str(round(mtr2.value()[0],2))+' '+str(round(mtr3.value()[0],2))+' '+str(round(mtr4.value()[0],2))+' '+str(round(mtr5.value()[0],2))
            valid_metrics = valid_fn(model, criterion, valid_loader, device, num_classes)
            ##############
            ## write events
            write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
            #save_weights(model, model_path, epoch + 1, step)
            #########################
            ## tensorboard
            write_tensorboard(writer, model, epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
            #########################
            ## save the best model
            valid_loss = valid_metrics['loss1']
            valid_jaccard = valid_metrics['jaccard']
            if valid_loss < previous_valid_loss:
                save_weights(model, model_path, epoch + 1, step, train_metrics, valid_metrics)
                previous_valid_loss = valid_loss
                print('Save best model by loss')
            if valid_jaccard > previous_valid_jaccard:
                save_weights(model, model_path, epoch + 1, step, train_metrics, valid_metrics)
                previous_valid_jaccard = valid_jaccard
                print('Save best model by jaccard')
            #########################
            ## change learning rate
            scheduler.step(valid_metrics['loss1'])

        except KeyboardInterrupt:
            # print('--' * 10)
            # print('Ctrl+C, saving snapshot')
            # save_weights(model, model_path, epoch, step)
            # print('done.')
            # print('--' * 10)
            writer.close()
            #return
    writer.close()




if __name__ == '__main__':
    # python train.py --image-path /media/eric/SSD2/Project/11_ISCB2018/0_Data/Task2/h5/
    # tensorboard --logdir runs --port 6008
    main()