import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

import core.config as config
import logger.meter as meter
import logger.logging as logging
import logger.checkpoint as checkpoint
from core.builder import setup_env
from core.config import cfg
from datasets.loader import get_normal_dataloader
from logger.meter import TrainMeter, TestMeter
from runner.criterion import KD_Loss

from net.resnet_c100 import resnet101
from net.resnet_c10 import resnet110
from net.repvgg_cifar import RepVGG_A1


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main():
    
    cycle_times = 0
    
    while True:
        setup_env()
        # Networks
        if cfg.LOADER.NUM_CLASSES == 100:
            teacher_net = resnet101().cuda()
            ckpt = torch.load('weights/resnet101.pth')
            teacher_net.load_state_dict(ckpt)
        elif cfg.LOADER.NUM_CLASSES == 10:
            teacher_net = resnet110().cuda()
            new_sd = OrderedDict()
            old_sd = torch.load('weights/resnet110_c10.th')['state_dict']
            for k,v in old_sd.items():
                if k.startswith('module.'):
                    new_sd[k[7:]] = v
                else:
                    new_sd[k] = v
            teacher_net.load_state_dict(new_sd)
        
        if cycle_times == 0:
            ckpt = torch.load(cfg.POST.PATH)
        else:
            ckpt = torch.load(os.path.join(cfg.OUT_DIR, "checkpoints", "cycle_"+str(cycle_times-1)+".pyth"))
            # lr_decay
            if hasattr(cfg.POST, "LR_DECAY"):
                cfg.OPTIM.BASE_LR = cfg.OPTIM.BASE_LR * (cfg.POST.LR_DECAY ** cycle_times)
                logger.info("Decayed LR for cycle {}: {}".format(cycle_times, cfg.OPTIM.BASE_LR))
            
        student_net = RepVGG_A1(num_classes=cfg.LOADER.NUM_CLASSES)
        student_net.load_state_dict(ckpt['model_state'])
        
        # rep branches into Rep_conv
        student_net._reparam(first=True)
        # inversion turn
        student_net.inverse_turn_all(1.)
        # student_net.postrep_turn(0.02)
        student_net.cuda()
        
        # Dataloaders
        [train_loader, valid_loader] = get_normal_dataloader()
        
        # Optim & Loss & LR
        # criterion = nn.CrossEntropyLoss()
        criterion = KD_Loss(alpha=cfg.POST.ALPHA, temperature=cfg.POST.TEMPERATURE)
        # criterion = DKD_Loss(alpha=cfg.DECO.ALPHA, temperature=cfg.DECO.TEMPERATURE)
        
        net_params = [
            {"params": student_net.weights(rep=False), "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
            {"params": student_net.weights(rep=True), "weight_decay": 0},
        ]
        optimizer = optim.SGD(net_params, cfg.OPTIM.BASE_LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.OPTIM.MAX_EPOCH)
        
        # Meters
        train_meter = TrainMeter(len(train_loader))
        test_meter = TestMeter(len(valid_loader))
        best_top1 = 100.1
        
        for cur_epoch in range(cfg.OPTIM.MAX_EPOCH):
            if cur_epoch > 0:
                student_net.set_attach_rate(min(((cur_epoch + 1.) / cfg.POST.WARMUP), 1.) * 1.)
            train_epoch(cur_epoch, teacher_net, student_net, train_loader, train_meter, optimizer, scheduler, criterion)
            if (cur_epoch + 1) % cfg.EVAL_PERIOD == 0 or (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH:
                top1err, top5err = test_epoch(cur_epoch, student_net, valid_loader, test_meter)
                if top1err < best_top1:
                    best_top1 = top1err
                    checkpoint.save_checkpoint(student_net, cur_epoch, best=True)
                    
                    # save model for cycling
                    sd = student_net.state_dict()
                    checkpoint_net = {
                        "epoch": cur_epoch,
                        "model_state": sd,
                    }
                    torch.save(checkpoint_net, os.path.join(cfg.OUT_DIR, "checkpoints", "cycle_"+str(cycle_times)+".pyth"))
        
        logger.info("Time_cycle:{} Best_top1:{}".format(cycle_times, test_meter.min_top1_err))
        cycle_times += 1
        
        if hasattr(cfg.POST, "CYCLE") and cycle_times >= cfg.POST.CYCLE:
            break
        

def train_epoch(cur_epoch, teacher_net, student_net, train_loader, train_meter, optimizer, scheduler, criterion):
    teacher_net.eval()
    student_net.train()
    
    lr = scheduler.get_last_lr()[0]
    cur_step = cur_epoch * len(train_loader)
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Forward
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        with torch.no_grad():
            raw_teacher_preds = teacher_net(inputs)
            teacher_preds = raw_teacher_preds.clone().detach()
        student_preds = student_net(inputs)
        loss = criterion(student_preds, teacher_preds, labels)
        # loss = criterion(student_preds, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student_net.weights(rep=False), cfg.OPTIM.GRAD_CLIP)
        
        if cur_epoch < cfg.POST.WARMUP:
            student_net.freeze_conv3_grad()
        
        optimizer.step()
        
        # Compute the errors
        top1_err, top5_err = meter.topk_errors(student_preds, labels, [1, 5])
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        cur_step += 1
    # Log epoch stats
    top1_err = train_meter.get_epoch_stats(cur_epoch)["top1_err"]
    top5_err = train_meter.get_epoch_stats(cur_epoch)["top5_err"]
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    scheduler.step()
    # Saving checkpoint
    if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
        checkpoint.save_checkpoint(student_net, cur_epoch, best=False)
    return top1_err, top5_err


@torch.no_grad()
def test_epoch(cur_epoch, net, test_loader, test_meter):
    net.eval()
    test_meter.reset()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = net(inputs)
        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()

        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # top1_err = test_meter.mb_top1_err.get_global_avg()
    # top5_err = test_meter.mb_top5_err.get_global_avg()
    top1_err = test_meter.get_epoch_stats(cur_epoch)["top1_err"]
    top5_err = test_meter.get_epoch_stats(cur_epoch)["top5_err"]
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    return top1_err, top5_err


if __name__ == '__main__':
    main()
    
    """Check cycle reparam"""
    
    # ckpt = torch.load(cfg.POST.PATH)
    # student_net = RepVGG_A1(num_classes=100)
    # student_net.load_state_dict(ckpt['model_state'])
    # student_net.cuda()
    # student_net.eval()
    
    # [train_loader, valid_loader] = get_normal_dataloader()
    
    # test_meter = TestMeter(len(valid_loader))
    # top1, top5 = test_epoch(0, student_net, valid_loader, test_meter)
    # print(top1, top5)
    
    # student_net._reparam(first=True)
    # test_meter.reset(True)
    # top1, top5 = test_epoch(0, student_net, valid_loader, test_meter)
    # print(top1, top5)
    
    # student_net.inverse_turn_all(0.)
    # test_meter.reset(True)
    # top1, top5 = test_epoch(0, student_net, valid_loader, test_meter)
    # print(top1, top5)
    
    # ckpt = torch.load("weights/repvgg.pyth")
    # net = RepVGG_A1(100)
    # net.load_state_dict(ckpt["model_state"])
    # net.eval()
    # x = torch.randn(1,3,32,32)
    
    # net.turn_only310(True)
    # print("train mode: {}".format(net(x)))

    # net._reparam()
    # print("rep mode: {}".format(net(x)))
    
    # net.init_branch_weights()
    # net.set_attach_rate(0.01)
    # net.turn_only310(False)
    # net._train()
    # print("post-rep mode: {}".format(net(x)))
    
    # net.turn_only310(False)
    # net = postrep_turn(net, 0.1)
    # print("post-rep mode: {}".format(net(x)))
