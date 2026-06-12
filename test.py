import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_frame
from engine.processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Testing")
    parser.add_argument(
        "--config_file", default="configs/RGBNT201/Signal.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=1, type=int)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    new_output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.ckpt_test_path)
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)

    logger = setup_logger("Signal", new_output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    model = make_frame(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.eval()
    '''
    RGBNT201:
    baseline : /media/zpp2/Datamy/lyy/612nw/6&12_bs
    baseline+SIM: 6&12_bs-sim-802
    basleine+SIM+cls:  6&12_bs-cls_0.2
    baseline+SIM+cls+pat: 6&12_bs-pat_0.2
    '''
    model.load_param(trained_path = "/media/zpp2/Datamy/lyy/signal_50.pth")
    do_inference(cfg, model, val_loader, num_query, logger, cfg.MODEL.stageName,args.local_rank)
