import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
import numpy as np
from datetime import datetime
from lib.Deeplab import Deeplab
from utils import dataset_deeplab as dataset
from utils.metrics import StreamSegMetrics
from utils.utils import clip_gradient, adjust_lr
from utils import transform as tr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

def train(train_loader, model, optimizer, epoch, save_path, writer, total_step):
    """
    train function
    """
    global step
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    loss_all = 0
    epoch_step = 0
    CEloss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.to(device, dtype=torch.float32)
            gts = gts.to(device, dtype=torch.long)

            preds, _, _ = model(images)

            loss = CEloss(preds, gts)

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data), flush=True)
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                    format(epoch, opt.epoch, i, total_step, loss.data))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.', flush=True)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!', flush=True)
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_accuracy, best_epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = StreamSegMetrics(10)
    metrics.reset()
    model.eval()
    correct_sample = 0
    total_sample = 0
    with torch.no_grad():
        for i, (image, gt) in enumerate(test_loader, start=1):
            image = image.to(device, dtype=torch.float32)
            gt = gt.to(device, dtype=torch.long)
            res, _, _ = model(image)
            preds = res.detach().max(dim=1)[1].cpu().numpy()
            targets = gt.cpu().numpy()
            metrics.update(targets, preds)

        score = metrics.get_results()
        mIou = score['Mean IoU']
        acc = score['Overall Acc']
        writer.add_scalar('Accuracy', torch.tensor(acc), global_step=epoch)
        writer.add_scalar('mIou', torch.tensor(mIou), global_step=epoch)
        if epoch == 1:
            best_accuracy = mIou
        else:
            if mIou > best_accuracy:
                best_accuracy = mIou
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch), flush=True)
        logging.info(
            '[Val Info]:Epoch:{} mIou:{} acc:{} bestEpoch:{} bestAccuracy:{}'.format(epoch, mIou, acc, best_epoch, best_accuracy))
        print('Epoch: {}, mIou: {:.4f}, acc: {:.4f}, bestAccuracy: {:.4f}, bestEpoch: {}.'.format(epoch, mIou, acc, best_accuracy, best_epoch), flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--val_batchsize', type=int, default=4, help='evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data_loader')
    parser.add_argument('--num_classes', type=int, default=10, help='number of iron types')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_root', type=str, default='./dataset', help='images root')
    parser.add_argument('--save_path', type=str, default='./savedModel/Deeplab/', help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0', flush=True)
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1', flush=True)
    cudnn.benchmark = True

    # build the model
    model = Deeplab(num_classes=opt.num_classes, backbone='mobilenet', output_stride=16,sync_bn=True, freeze_bn=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {str(total_params)[:-6] + '.' + str(total_params)[-6:-4] + 'M'}", flush=True)

    model = model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load, flush=True)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    train_transform = tr.ExtCompose([
            # et.ExtResize( 512 ),
            tr.ExtRandomCrop(size=(513, 513)),
            tr.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            tr.ExtRandomHorizontalFlip(),
            tr.ExtToTensor(),
            tr.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    val_transform = tr.ExtCompose([
        #tr.ExtResize( 513 ),
        tr.ExtToTensor(),
        tr.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_sampler = None
    val_sampler = None

    # load data
    print('Loading data...', flush=True)
    train_data = dataset.SemData(split='train', data_root=opt.data_root, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchsize, shuffle=(train_sampler is None), num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last=True)
    val_data = dataset.SemData(split='eval', data_root=opt.data_root, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.val_batchsize, shuffle=False, num_workers=opt.num_workers, pin_memory=True, sampler=None)
    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0
    total_step = len(train_loader)

    print("Start train...", flush=True)
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer, total_step)
        val(val_loader, model, epoch, save_path, writer)
