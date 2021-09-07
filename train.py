######################## train YOLOv3 example ########################
import os
import argparse
import ast
from easydict import EasyDict as edict
import shutil

import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from mindspore.common import set_seed

from src.yolov3 import yolov3_resnet18, YoloWithLossCell, TrainingWrapper
from src.dataset import create_yolo_dataset, data_to_mindrecord_byte_image
from src.config import ConfigYOLOV3ResNet18

import moxing as mox

set_seed(1)

sys.path.append('..')

def get_lr(learning_rate, start_step, global_step, decay_step, decay_rate, steps=False):
    """Set learning rate."""
    lr_each_step = []
    for i in range(global_step):
        if steps:
            lr_each_step.append(learning_rate * (decay_rate ** (i // decay_step)))
        else:
            lr_each_step.append(learning_rate * (decay_rate ** (i / decay_step)))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = lr_each_step[start_step:]
    return lr_each_step


def init_net_param(network, init_value='ones'):
    """Init the parameters in network."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_data(initializer(init_value, p.data.shape, p.data.dtype))


def main(args_opt):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    if args_opt.distribute:
        device_num = args_opt.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank = args_opt.device_id % device_num
    else:
        rank = 0
        device_num = 1

    loss_scale = float(args_opt.loss_scale)
    
    # When create MindDataset, using the fitst mindrecord file, such as yolo.mindrecord0.
    dataset = create_yolo_dataset(args_opt.mindrecord_file,
                                  batch_size=args_opt.batch_size, device_num=device_num, rank=rank)
    dataset_size = dataset.get_dataset_size()
    print('The epoch size: ', dataset_size)
    print("Create dataset done!")

    net = yolov3_resnet18(ConfigYOLOV3ResNet18())
    net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
    init_net_param(net, "XavierUniform")

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs,
                                  keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=cfg.ckpt_dir, config=ckpt_config)

    if args_opt.pre_trained:
        if args_opt.pre_trained_epoch_size <= 0:
            raise KeyError("pre_trained_epoch_size must be greater than 0.")
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    total_epoch_size = 1
    if args_opt.distribute:
        total_epoch_size = 1
    lr = Tensor(get_lr(learning_rate=args_opt.lr, start_step=args_opt.pre_trained_epoch_size * dataset_size,
                       global_step=total_epoch_size * dataset_size,
                       decay_step=1000, decay_rate=0.95, steps=True))
    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)
    
    callback = [LossMonitor(10*dataset_size), ckpoint_cb]
    model = Model(net)
    dataset_sink_mode = cfg.dataset_sink_mode
    print("Start train YOLOv3, the first epoch will be slower because of the graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)
# ------------yolov3 train -----------------------------
cfg = edict({
    "distribute": False,
    "device_id": 0,
    "device_num": 1,
    "dataset_sink_mode": True,

    "lr": 0.001,
    "epoch_size": 1,
    "batch_size": 32,
    "loss_scale" : 1024,

    "pre_trained": None,
    "pre_trained_epoch_size":0,

    "ckpt_dir": "./ckpt",
    "save_checkpoint_epochs" :1,
    'keep_checkpoint_max': 1,

    "data_url": 's3://yyq-2/DATA/code/yolov3/mask_detection_500',
    "train_url": 's3://yyq-2/DATA/code/yolov3/yolov3_out/',
    "label_url": 'obs://mask-detection-hong-kong-bj4/MSPDC/dataset/label',
    'metadata_url': '',
}) 

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Yolo Training')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')   
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--label_url', required=True, default=None, help='Location of data for labels.')
    parser.add_argument('--metadata_url', required=True, default=None, help='Location of data for metadata.')
    parser.add_argument('--dataset_url', required=True, default=None, help='Location of data for images.')

    args_opt = parser.parse_args()
    args_opt.data_url = args_opt.dataset_url

    if args_opt.data_url[:5] != 'obs:/' and args_opt.data_url[:4] != 's3:/':
        args_opt.data_url = 'obs:/' + args_opt.data_url
    cfg.data_url = args_opt.data_url
    
    if args_opt.train_url[:5] != 'obs:/' and args_opt.train_url[:4] != 's3:/':
        args_opt.train_url = 'obs:/' + args_opt.train_url
    cfg.train_url = args_opt.train_url

    if args_opt.label_url[:5] != 'obs:/' and args_opt.label_url[:4] != 's3:/':
        args_opt.label_url = 'obs:/' + args_opt.label_url
    cfg.label_url = args_opt.label_url
    
    if args_opt.metadata_url[:5] != 'obs:/' and args_opt.metadata_url[:4] != 's3:/':
        args_opt.metadata_url = 'obs:/' + args_opt.metadata_url
    cfg.metadata_url = args_opt.metadata_url

    if os.path.exists(cfg.ckpt_dir):
        shutil.rmtree(cfg.ckpt_dir)
    data_path = './data/' 
    if not os.path.exists(data_path):
        mox.file.copy_parallel(src_url=cfg.data_url, dst_url=data_path)
    label_path = './label.csv' 
    if not os.path.exists(label_path):
        mox.file.copy_parallel(src_url=cfg.label_url, dst_url=label_path)
    metadata_path = './metadata.csv' 
    if not os.path.exists(metadata_path):
        mox.file.copy_parallel(src_url=cfg.metadata_url, dst_url=metadata_path)

    mindrecord_dir_train = os.path.join(data_path,'mindrecord/train')

    print("Start create dataset!")
    # It will generate mindrecord file in args_opt.mindrecord_dir,and the file name is yolo.mindrecord.
    prefix = "yolo.mindrecord"
    cfg.mindrecord_file = os.path.join(mindrecord_dir_train, prefix)
    if os.path.exists(mindrecord_dir_train):
        print('The mindrecord file had exists!')
    else:
        image_dir = os.path.join(data_path, 'images')
        if not os.path.exists(mindrecord_dir_train):
            os.makedirs(mindrecord_dir_train)
        print("Create Mindrecord.")
        data_to_mindrecord_byte_image(image_dir, mindrecord_dir_train, prefix, 1, label_path, metadata_path, train_test_split=0.99)
        print("Create Mindrecord Done, at {}".format(mindrecord_dir_train))
        # if you need use mindrecord file next time, you can save them to yours obs.
        #mox.file.copy_parallel(src_url=args_opt.mindrecord_dir_train, dst_url=os.path.join(cfg.data_url,'mindspore/train')

    main(cfg)
    mox.file.copy_parallel(src_url=cfg.ckpt_dir, dst_url=cfg.train_url)
