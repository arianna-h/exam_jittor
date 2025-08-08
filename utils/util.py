import jittor as jt
from jittor import nn
from datetime import datetime
import numpy as np
def clip(tensor, min_val, max_val):
    return jt.minimum(jt.maximum(tensor, min_val), max_val)


def bbox_iou(bbox_a, bbox_b):
    #if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
    if hasattr(bbox_a, 'numpy'):
        bbox_a = bbox_a.numpy()
    if hasattr(bbox_b, 'numpy'):
        bbox_b = bbox_b.numpy()
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox2loc(src_bbox, dst_bbox):
    #确保输入是Jittor张量
    if not isinstance(src_bbox, jt.Var):
        src_bbox = jt.array(src_bbox)
    if not isinstance(dst_bbox, jt.Var):
        dst_bbox = jt.array(dst_bbox)
    
        #防止越界
    widths = jt.maximum(src_bbox[:, 2] - src_bbox[:, 0], 1.0)
    heights = jt.maximum(src_bbox[:, 3] - src_bbox[:, 1], 1.0)
    ctr_x = src_bbox[:, 0] + 0.5 * widths
    ctr_y = src_bbox[:, 1] + 0.5 * heights

    #print(dst_bbox.shape)
    base_widths = jt.maximum(dst_bbox[:, 2] - dst_bbox[:, 0], 1.0)
    base_heights = jt.maximum(dst_bbox[:, 3] - dst_bbox[:, 1], 1.0)
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_widths
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_heights
    
    dx = (base_ctr_x - ctr_x) / widths #计算偏移量
    dy = (base_ctr_y - ctr_y) / heights
    dw = jt.log(jt.maximum(base_widths / widths, 1e-5))
    dh = jt.log(jt.maximum(base_heights / heights, 1e-5))
    
    return jt.stack([dx, dy, dw, dh], dim=1)



def loc2bbox(src_bbox, loc): #[N, 4]    #[N, 4 * num_classes]
    if src_bbox.size()[0] == 0:
        return jt.zeros((0, 4), dtype=loc.dtype)

    src_width   = jt.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)  #宽
    src_height  = jt.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)  #高
    src_ctr_x   = jt.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width  #中心x
    src_ctr_y   = jt.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height  # y

    dx          = loc[:, 0::4]  #各偏移
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]

    #得到bbox
    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = jt.exp(dw) * src_width
    h = jt.exp(dh) * src_height

    # xy xy 
    dst_bbox = jt.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox #n 类的box  [n_proposal,4]



def log_training_status(global_step, epoch, total_loss, dect_losses, rpn_losses, lr,log_path):
    """
    记录训练损失日志（单行格式
    """

    dect_box_val, dect_cls_val= [x.item() for x in dect_losses]
    rpn_box_val, rpn_cls_val= [x.item() for x in rpn_losses]

    #模块总loss
    dect_sum=dect_box_val+dect_cls_val
    rpn_sum=rpn_box_val+rpn_cls_val

    #格式化输出字符串
    dect_box=f"{dect_box_val:.4f}"
    dect_cls =f"{dect_cls_val:.4f}"
    dect_total =f"{dect_sum:.4f}"

    rpn_box= f"{rpn_box_val:.4f}"
    rpn_cls= f"{rpn_cls_val:.4f}"
    rpn_total= f"{rpn_sum:.4f}"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = (
        f"[{now} train.log] INFO: Epoch: [{epoch}]  [Step {global_step}]  "
        f"lr: {lr:.6f}  "
        f"loss: {total_loss.item():.4f}  "
        f"detection_loss: {dect_total} (cls: {dect_cls}, box: {dect_box})  "
        f"rpn_loss: {rpn_total} (cls: {rpn_cls}, box: {rpn_box})\n"
    )
    
    print(log_line.strip())  # 控制台输出（可选）
    with open(log_path,'a') as f:
        f.write(log_line)



def format_coco_stats(stats):
    names = [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
    ]
    res = "IoU metric: bbox\n"
    for name, val in zip(names, stats):
        res += f"{name} = {val:.3f}\n"
    return res


def format_per_class_results(coco_eval, cat_id_to_name):
    precisions = coco_eval.eval['precision']  # shape: [T, R, K, A, M]
    recalls = coco_eval.eval['recall']        # shape: [T, K, A, M]
    cat_ids = coco_eval.params.catIds

    ap_per_class = []

    for i, cat_id in enumerate(cat_ids):
        name = cat_id_to_name[cat_id]
        # AP: mean over IoU=0.5:0.95, area=all, maxDets=100
        precision = precisions[:, :, i, 0, 2]
        valid = precision[precision > -1]
        ap = valid.mean() if valid.size else float('nan')

        # Recall: IoU=0.5:0.95, area=all, maxDets=100
        recall = recalls[:, i, 0, 2]
        valid = recall[recall > -1]
        rec = valid.mean() if valid.size else float('nan')

        # GT/Dets
        gts = coco_eval.evalImgs[i::len(cat_ids)]
        gt_num = sum([gi['gtIds'].__len__() for gi in gts if gi is not None])
        det_num = sum([gi['dtIds'].__len__() for gi in gts if gi is not None])

        ap_per_class.append((name, gt_num, det_num, rec, ap))

    # 排序按类别名
    ap_per_class.sort(key=lambda x: x[0])
    lines = []
    lines.append("+--------------+------+-------+--------+-------+")
    lines.append("| class        | gts  | dets  | recall | ap    |")
    lines.append("+--------------+------+-------+--------+-------+")
    for name, gts, dets, rec, ap in ap_per_class:
        lines.append(f"| {name:<12} | {gts:<4} | {dets:<5} | {rec:.3f}  | {ap:.3f} |")
    lines.append("+--------------+------+-------+--------+-------+")

    # mean
    mean_ap = np.mean([x[-1] for x in ap_per_class if not np.isnan(x[-1])])
    mean_rec = np.mean([x[-2] for x in ap_per_class if not np.isnan(x[-2])])
    lines.append(f"| mean results | ---- | ---- | {mean_rec:.3f}  | {mean_ap:.3f} |")
    lines.append("+--------------+------+-------+--------+-------+")

    return "\n".join(lines)





# anchor匹配样本
class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample       = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio      = pos_ratio

    def __call__(self, bbox, anchor): # all ,4            M,4
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any(): # arg max iou 值每个框对应的gt
            loc = bbox2loc(anchor, bbox[argmax_ious]) #bbox[argmax_ious] 取出对应的gt与anchor 映射

            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        #----------------------------------------------#
        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        #----------------------------------------------#

        ious = bbox_iou(anchor, bbox)
       # print(f"ious in calc {ious.shape}") # all  M
        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        #---------------------------------------------------------#
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        #---------------------------------------------------------#
      #  ious_np = ious.numpy()
        argmax_ious ,max_ious= jt.argmax(ious,dim=1)  # 每个anchor 对应的gt
        #argmax_ious = ious.argmax(axis=1)
        #---------------------------------------------------------#
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        #---------------------------------------------------------#
      #  max_ious = np.max(ious_np, axis=1) #每个anchor 的最大iou
        #---------------------------------------------------------#
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        #---------------------------------------------------------#
        gt_argmax_ious,_ = jt.argmax(ious,dim=0)  #每个gt对应的anchor
        #---------------------------------------------------------#
        #   保证每一个真实框都存在对应的先验框
        #---------------------------------------------------------#
     #   print(gt_argmax_ious)
     #   print(ious)
     #   print(argmax_ious)

   
     #   print("flag")
        for i in range(len(gt_argmax_ious)): 
            argmax_ious[gt_argmax_ious[i]] = i   #遍历对应gt的最大iou anchor  将其设置维对应label  M 

        return argmax_ious, max_ious, gt_argmax_ious
        
    def _create_label(self, anchor, bbox):
        # ------------------------------------------ #
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        # ------------------------------------------ #
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # ------------------------------------------------------------------------ #
        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        # ------------------------------------------------------------------------ #
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        # argmax iou 就是anchro对应的标签
        # ----------------------------------------------------- #
        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        # ----------------------------------------------------- #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious)>0:
            label[gt_argmax_ious] = 1 # 把gt对应的框设置维1

        # ----------------------------------------------------- #
        #   判断正样本数量是否大于128，如果大于则限制在128
        # ----------------------------------------------------- #
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # ----------------------------------------------------- #
        #   平衡正负样本，保持总数量为256
        # ----------------------------------------------------- #
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False) #其他非正非负设置为-1
            label[disable_index] = -1

        return argmax_ious, label


#匹配正负样本 然后采样进行检测器训练
class ProposalTargetCreator(object):
    def __init__(self, n_sample=256, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)): 



        roi = np.concatenate((roi.numpy(), bbox),axis=0)  # 把gt加入roi  3000+M , 4

        iou = bbox_iou(roi, bbox)  # 3000+m , M

        if len(bbox)==0:
            print("????")
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:

            gt_assignment = iou.argmax(axis=1)
            max_iou = iou.max(axis=1)
            # 真实框的标签要+1因为有背景的存在
            gt_roi_label = label[gt_assignment] + 1

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)


        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
                   
        #---------------------------------------------------------#
        #   sample_roi      [n_sample, 4]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]]) #计算偏移并且标准化
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        
        gt_roi_label[pos_roi_per_this_image:] = 0  #后面负样本类别是 0

      #  print(f"检测器阶段 正样本数目 {pos_roi_per_this_image}")
      #  print(gt_roi_label)
    #    print(f"proposal assign sample roi{ sample_roi.shape}")
     #   print(f"proposal assign gt_roi_loc{ gt_roi_loc.shape}")
    #    print(f"proposal assign gt_roi_label{ gt_roi_label.shape}")
        return sample_roi, gt_roi_loc, gt_roi_label






def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
from PIL import Image

#对输入图像进行resize
def resize_image(image, size):
    w, h        = size
    new_image   = image.resize((w, h), Image.BICUBIC)
    return new_image

# 获得类
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def preprocess_input(image):
    image /= 255.0
    return image

def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr_epoch0 = 1e-5       # epoch 0
        self.lr_epoch1 = 5e-5       # epoch 1
        self.lr_epoch2 = 1e-4       # epoch 2
        self.lr_final   = 1e-6      # 最终学习率
        self.total_epochs = 30
        self.decay_start_epoch = 3

    def get_lr(self, epoch):
        epoch = epoch % self.total_epochs
        if epoch == 0:
            return self.lr_epoch0
        elif epoch == 1:
            return self.lr_epoch1
        elif epoch == 2:
            return self.lr_epoch2
        elif epoch >= 3 and epoch < self.total_epochs:
            decay_epochs = self.total_epochs - self.decay_start_epoch - 1
            current_decay_epoch = epoch - self.decay_start_epoch
            return self.lr_epoch2 - (self.lr_epoch2 - self.lr_final) * (current_decay_epoch / decay_epochs)
        else:
            return self.lr_final

    def step(self, epoch):
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train):
        super(FasterRCNNTrainer, self).__init__()
        self.model_train    = model_train
        self.rpn_sigma      = 1
        self.roi_sigma      = 1

        self.anchor_target_creator      = AnchorTargetCreator()
        self.proposal_target_creator    = ProposalTargetCreator()

        self.loc_normalize_std          = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc    = pred_loc[gt_label > 0]
        gt_loc      = gt_loc[gt_label > 0]
        #取出 正样本计算
        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = jt.where(
                regression_diff < (1. / sigma_squared),
                0.5 * sigma_squared * regression_diff ** 2,
                regression_diff - 0.5 / sigma_squared
            )
        regression_loss = regression_loss.sum()
        num_pos         = (gt_label > 0).sum().float()
        #避免除以0
        regression_loss /= jt.maximum(num_pos, jt.ones_like(num_pos))

        return regression_loss
        
    def forward(self, imgs, bboxes, labels):
        n           = imgs.shape[0]
        img_size    = imgs.shape[2:]
        #-------------------------------#
        #   获取公用特征层
        #-------------------------------#
        base_feature = self.model_train(imgs, mode = 'extractor')

        # -------------------------------------------------- #
        #   利用rpn网络获得调整参数、得分、建议框、先验框
        # -------------------------------------------------- #
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(x = [base_feature, img_size], mode = 'rpn')
        
        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all  = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels                 = [], [], [], []
        for i in range(n):
            bbox        = bboxes[i] # M,4
            label       = labels[i] # M
            rpn_loc     = rpn_locs[i] #all ,4
            rpn_score   = rpn_scores[i] # all ,2   anchor [all,4]
            roi         = rois[i] # 3000,4 

        #    print(f"bbox {bbox.shape}")
          #  print(f"label {label.shape}")
           # print(f"rpn_loc {rpn_loc.shape}")
           # print(f" rpn_score { rpn_score.shape}")
           # print(f"roi {roi.shape}")
           # print(f"anchor {anchor.shape}")
            # -------------------------------------------------- #
            #   利用真实框和先验框获得建议框网络应该有的预测结果
            #   给每个先验框都打上标签
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label    = self.anchor_target_creator(bbox, anchor[0])   #?
            #得到  一张图片对应的gt 偏移 gt_rpn_loc 和 lablegt_rpn_label和上面shape一样


            gt_rpn_loc = jt.array(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label = jt.array(gt_rpn_label).type_as(rpn_locs).long()
            # -------------------------------------------------- #
            #   分别计算建议框网络的回归损失和分类损失
            # -------------------------------------------------- #
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma) # 回归损失
       #     print("前景 (1) 个数:", (gt_rpn_label == 1).sum())
         #   print("背景 (0) 个数:", (gt_rpn_label == 0).sum())
         #   print("忽略 (-1) 个数:", (gt_rpn_label == -1).sum())
            rpn_cls_loss = nn.cross_entropy_loss(rpn_score, gt_rpn_label, ignore_index=-1) #由于在分配就 忽略-1 并且正负样本均匀了
  
            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            # ------------------------------------------------------ #
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   获得三个变量，分别是sample_roi, gt_roi_loc, gt_roi_label
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            # ------------------------------------------------------ #
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)

            sample_rois.append(sample_roi)
            sample_indexes.append(jt.ones([len(sample_roi),]) * roi_indices[i][0])
            gt_roi_locs.append(gt_roi_loc)
            gt_roi_labels.append(gt_roi_label)  # 一般 label 用 long（int64）


        sample_rois     = jt.stack(sample_rois, dim=0) #形成一个大batch 传入处理
        sample_indexes  = jt.stack(sample_indexes, dim=0) 

        roi_cls_locs, roi_scores = self.model_train([base_feature, sample_rois, sample_indexes, img_size], mode = 'head')
       # [B, num , cls ]   [B num cls*4]

        for i in range(n):
            # ------------------------------------------------------ #
            #   根据建议框的种类，取出对应的回归预测结果
 
            n_sample =roi_cls_locs[i].size(0)

            roi_cls_loc     = roi_cls_locs[i] #[B ,num ,cls*4]
            roi_score       = roi_scores[i] #[B, num , cls ]
            gt_roi_loc      = gt_roi_locs[i] # B num 4  真实样本的偏移
            gt_roi_label    = gt_roi_labels[i] # b num ,

            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)  # num , cls , 4
       #     print(roi_cls_loc.shape)
            roi_loc     = roi_cls_loc[jt.arange(n_sample), gt_roi_label]   #获取lable 对应的预测偏移  num,4

            # -------------------------------------------------- #
            #   分别计算Classifier网络的回归损失和分类损失
            # -------------------------------------------------- #
          #  print(gt_roi_label)
        #    print(gt_roi_loc)
        #    print(roi_loc)
        #    print(gt_roi_labels)
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
            roi_cls_loss = nn.cross_entropy_loss(roi_score, gt_roi_label)

            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss
            
        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        losses = losses + [sum(losses)]

        return losses

    def train_step(self, imgs, bboxes, labels):
        

        # 前向传播得到各个 loss，最后一个是 total_loss
        losses = self.forward(imgs, bboxes, labels)


        return losses



