import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
import math
from layers.multimodal_memory import MultiModalClusterMemory
from layers.memory_utils import extract_multimodal_features

def adjust_weights(epoch, total_epochs, initial_weight, final_weight):
    alpha = min(epoch / (total_epochs * 0.2), 1.0)
    return initial_weight + alpha * (final_weight - initial_weight)

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_loader_normal,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, stage):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    alpha = cfg.MODEL.Gram_Loss_weight
    beta = cfg.MODEL.PAT_Loss_weight
    mmc_weight = cfg.MODEL.MMC_LOSS_WEIGHT
    memory_momentum = cfg.MODEL.MEMORY_MOMENTUM
    memory_temp = cfg.MODEL.MEMORY_TEMP
    use_mmc = cfg.MODEL.USE_MMC

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("Signal.train")
    logger.info('start training')

    if device:
        print(f"use CUDA {device}")
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device],
                                                              find_unused_parameters=True)
          
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler() 
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Start Training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()

        # ---- MMC: initialise 3-modal memory bank at the start of each epoch ----
        mmc_memory = None
        if use_mmc:
            _m = model.module if hasattr(model, 'module') else model
            feat_dim_for_mem = _m.feat_dim

            rgb_feats, ni_feats, ti_feats, mem_labels = extract_multimodal_features(
                model, train_loader_normal, device
            )
            rgb_feats = F.normalize(rgb_feats.float(), dim=1)
            ni_feats  = F.normalize(ni_feats.float(), dim=1)
            ti_feats  = F.normalize(ti_feats.float(), dim=1)

            mem_num_classes = len(mem_labels.unique()) - 1 if -1 in mem_labels else len(mem_labels.unique())

            mmc_memory = MultiModalClusterMemory(
                num_classes=mem_num_classes,
                feat_dim=feat_dim_for_mem,
                temp=memory_temp,
                momentum=memory_momentum,
            )
            mmc_memory.set_features(rgb_feats, ni_feats, ti_feats, mem_labels, device=device)
            logger.info(f'MMC memory bank created: {mmc_memory.memory_rgb.features.shape[0]} proxies '
                        f'for {mem_num_classes} classes')
            model.train()  # restore training mode after feature extraction

        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()


            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            
            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view,training=True,sge=stage)

                loss = 0
                sign = output[0]

                if sign == 1:
                    index = len(output) - 1
                    for i in range(1, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
                    
                  
                elif sign == 2:
                    index = len(output) - 1
                    for i in range(1, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp

        
                else:
                    
                    if stage == "CLS":
                        index = len(output) - 2
                        for i in range(1, index, 2):
                            loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                            loss = loss + loss_tmp

                        CLS_loss = output[-1]
                        loss = loss + alpha * CLS_loss  

                    else:
                        index = len(output) - 3
                        for i in range(1, index, 2):
                            loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                            loss = loss + loss_tmp

                        CLS_loss,pat_loss = output[-2],output[-1]

                        loss = loss + alpha * CLS_loss + beta * pat_loss


            # ---- MMC loss (multi-modal memory collaboration) ----
            if use_mmc and mmc_memory is not None:
                # extract per-modality global features from model cache
                _m = model.module if hasattr(model, 'module') else model
                rgb_g = _m.rgb_global_feat
                ni_g  = _m.ni_global_feat
                ti_g  = _m.ti_global_feat
                loss_intra, loss_cross, loss_mmc = mmc_memory(rgb_g, ni_g, ti_g, target)
                loss = loss + mmc_weight * loss_mmc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()


            if isinstance(output, list):
                acc = (output[1][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[1].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))



        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        
        new_output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.ckpt_save_path)
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)
        
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(new_output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(new_output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,sge=stage)
                

                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                               os.path.join(new_output_dir, cfg.MODEL.NAME + 'best.pth'))
                logger.info("~" * 50)
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 logger,
                 sge,
                 local_rank):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("Signal.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        # if torch.cuda.device_count() > 1:
        #     print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        #     model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    logger.info("~" * 50)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view,training=False,sge=sge)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print("Validation Results ")
    print("mAP: {:.1%}".format(mAP))
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1,sge="CLS"):
    evaluator.reset()
    model.eval()
    logger.info("~" * 50)

    if not cfg.MODEL.USE_A:
        logger.info("Current is the base feature testing!")
    
    else:
        logger.info("Current is the our feature testing!")

    
    

    logger.info("~" * 50)

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view,return_pattern=return_pattern,training=False,sge=sge)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    torch.cuda.empty_cache()
    return mAP, cmc
