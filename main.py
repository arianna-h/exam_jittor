import os
import datetime
import jittor as jt
from jittor import nn
import numpy as np

from model.Faster_RCNN import Faster_RCNN
from utils.util import log_training_status,LRScheduler,FasterRCNNTrainer,get_classes
from dataset.data import FRCNNDataset
from utils.callback import EvalCallback
#----------------------------------------------------------------------#
#训练设置

#环境类
jt.flags.use_cuda = 1  #jt的CUDA训练
jt.set_global_seed(42) #种子



#训练设置类

batch_size = 10  #bs
num_epochs = 30  #epoch
lr = 1e-4        #初始lr
backbone='resnet'


#数据集类
num_classes=21 #总类别 + 1 背景类
img_path="./VOCdevkit/VOC2012/JPEGImages/" #训练数据文件夹

train_annotation_path="./pascal_anno/train.txt" #训练集划分
val_annotation_path="./pascal_anno/val.txt"
with open(train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()
train_lines = [img_path + line.strip() + "\n" for line in train_lines]

with open(val_annotation_path, encoding='utf-8') as f:
    val_lines   = f.readlines()
val_lines = [img_path + line.strip() + "\n" for line in train_lines]



#自动生成log路径并重定向所有输出
base_dir = os.path.dirname(os.path.abspath(__file__))
record_path = os.path.join(base_dir,"checkpoints/voc/",f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
os.makedirs(record_path, exist_ok=True)
log_file=os.path.join(record_path,'log.txt')
save = os.path.join(record_path,"checkpoints")
class_names, num_classes = get_classes("./model_data/voc_classes.txt")



model_path='' #没有预训练权重从零开始

model = Faster_RCNN(num_classes=num_classes, 
                    backbone=backbone,
                    anchor_scales   = [8, 16, 32], #anchor 在特征图的尺寸 ，如果缩放回原始图像会 是16倍大小
                    pretrained=False,
                    n_train_pre_nms=6000, #训练时 nms 前保留多少 proposal
                    n_train_post_nms=1000, # 训练时 nms后保留多少
                    nms_iou=0.7, # nms处理的iou
                    n_test_pre_nms=3000, # 预测时nms前保留多少
                    n_test_post_nms=600 #预测时nms后保留多少
                    )

#预训练权重进行加载
if model_path != '':
    print('Load weights {}.'.format(model_path))
    model_dict      = model.state_dict()
    pretrained_dict = jt.load(model_path)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

model.freeze_bn()


optimizer = nn.Adam(model.parameters(), lr=lr,betas = (0.9, 0.999),weight_decay=0)

scheduler = LRScheduler(optimizer)

trainer=FasterRCNNTrainer(model)
eval_callback   = EvalCallback(model, [600, 600], class_names, 21, val_lines, record_path, 
                                        eval_flag=1, period=1)

#由于自定义的数据接口太占显存，使用了pytorch版本的dataset接口
td=FRCNNDataset(annotation_lines=train_lines)
train_loader = jt.dataset.DataLoader(td, batch_size=batch_size, shuffle=False)
total_len=len(train_loader.dataset)




for epoch in range(num_epochs):
    global_step = 0
    model.train()
    save_path=os.path.join(save,f'{epoch}')
 
    current_lr=scheduler.step(epoch)
    for batch_idx,(images,gt_boxes,gt_labels)in enumerate(train_loader):

        optimizer.zero_grad()
        rpn_loc_loss,rpn_cls_loss,roi_loc_loss,roi_cls_loss,total_loss=trainer.train_step(images, gt_boxes, gt_labels)

        #optimizer.clip_grad_norm(max_norm=1.0, norm_type=2)  #

        optimizer.step(total_loss)


        global_step+= batch_size

        if global_step%100==0 or global_step==total_len:

            log_training_status(f"[{global_step}/{total_len}]",epoch, total_loss,[roi_loc_loss, roi_cls_loss],[rpn_loc_loss, rpn_cls_loss,],current_lr ,log_file)

 


    model.eval()
    eval_callback.on_epoch_end(epoch + 1)
    os.makedirs(save_path, exist_ok=True)

    model_filename=f"{epoch}_mAP.pkl"

    model_path= os.path.join(save_path, model_filename)
    jt.save(model.state_dict(), model_path)