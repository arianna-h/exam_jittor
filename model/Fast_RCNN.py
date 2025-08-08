import math
import jittor as jt
from jittor import nn
import jittor.init as init

from jdet.ops.roi_align import ROIAlign
from jdet.ops.roi_pool import ROIPool



class Fast_RCNN(nn.Module):
    def __init__(self, n_class, in_channel,roi_size, spatial_scale, classifier):
        super(Fast_RCNN, self).__init__()
        self.classifier = classifier
        #   对ROIPooling后的的结果进行回归预测

        self.cls_loc = nn.Linear(in_channel, n_class * 4)
        #   对ROIPooling后的的结果进行分类

        self.score = nn.Linear(in_channel, n_class)
        #-----------------------------------#

        self.ratio=spatial_scale

        # roi 特征提取器
        self.roi_op= ROIAlign(output_size=(roi_size[0],roi_size[1]), spatial_scale=1)


        init.gauss_(self.cls_loc.weight, mean=0.0, std=0.001)
        init.constant_(self.cls_loc.bias, 0.0)
        init.gauss_(self.score.weight, std=0.01)
        pi = 0.01
        init.constant_(self.score.bias, -math.log((1 - pi) / pi))

    def execute(
        self,
        features,                  # [B, C, H, W]
        rois,                      # list of [整个batch的总roi, 4]
        roi_idx,                  # list of [属于哪个图片]
        img_size  # H W
    ):
        
        B, _, H_feat, W_feat = features.shape
    
        rois        = jt.flatten(rois, 0, 1)
        roi_idx = jt.flatten(roi_idx, 0, 1)

        # roi_map 是坐标，float32
        roi_map = jt.zeros_like(rois).float32()
        roi_map[:, [0, 2]] = rois[:, 0:2] / img_size[1] * W_feat
        roi_map[:, [1, 3]] = rois[:, 1:3] / img_size[0] * H_feat

        # 转换 roi_i 为 float32 （数值不变，只是类型变成 float32）
       
        indices_and_rois = jt.concat([roi_idx[:, None], roi_map], dim=1)
        # 拼接，最终整体是 float32，符合 roi_op 的要求

        # 传入 roi_op
        pool = self.roi_op(features, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        #pool = self.roi_op(features,roi_pos )

        fc=self.classifier(pool)  #两个线性和 激活

       # fc =fc.view(fc.size(0),-1) # 分为batch flatten

        box= self.cls_loc(fc)  # b*N 4*cls

        cls = self.score(fc)   # b*N cls 
        del fc
        roi_cls_locs    = box.view(B, -1,box.size(1))
        roi_scores      = cls.view(B, -1,cls.size(1))
        return roi_cls_locs, roi_scores

'''
class Fast_RCNN_o(nn.Module):
    def __init__(self, in_features, num_classes,roi_mode='pooling',roi_size=(7,7),stride=16):
        super().__init__()
        self.stride=stride
        self.roi_mode=roi_mode
        #mode pooling align
        self.roi_op= ROIPool(output_size=7, spatial_scale=0.0625)
        self.fc1 = nn.Linear(in_features*roi_size[0]*roi_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024,  num_classes*4)
        self.init_weights()

    def execute(self, features, proposals, ratio=16, labels=None):
        batch_rois = []
        batch_offsets = [0]
        _, _, H_feat, W_feat = features.shape
        image_height = H_feat * ratio
        image_width = W_feat * ratio
        print(f"post size{len(proposals[0][0])}")
        # 合并所有batch的proposal roi
        for i in range(len(proposals)):
            anchor, box = proposals[i]
            roi = decode_boxes(anchor, box, image_width, image_height)   # [N_i, 4]
            batch_rois.append(roi)
            batch_offsets.append(batch_offsets[-1] + roi.shape[0])
        batch_rois = jt.concat(batch_rois, dim=0)   # [sum(N_i), 4]

        # batch indices
        batch_indices = []
        for i in range(len(proposals)):
            n = proposals[i][0].shape[0]
            batch_indices.append(jt.full((n,1), i))
        batch_indices = jt.concat(batch_indices, dim=0)  # [sum(N_i), 1]

        batch_rois_with_index = jt.concat([batch_indices, batch_rois], dim=1)  # [sum(N_i), 5]

        # roi pooling
        x = self.roi_op(features, batch_rois_with_index)  # [sum(N_i), C, pooled_h, pooled_w]

        x = x.flatten(1)
      #  x = nn.relu(self.fc1(x))
      #  x = nn.relu(self.fc2(x))
        cls_scores = self.cls_score(x)                     # [sum(N_i), num_classes]
        bbox_preds = self.bbox_pred(x)                     # [sum(N_i), num_classes*4]

        # reshape bbox_preds -> [N, num_classes, 4]
        N = bbox_preds.shape[0]
        bbox_preds = bbox_preds.reshape(N, -1, 4)

        # 预测类别
        if labels is None:
            # 推理时用最高分类别索引
            labels,_ = jt.argmax(cls_scores,dim=1)
        else:
            # 训练时保证labels是LongTensor且长度N
            labels = labels.reshape(-1)

        # 选择对应类别的bbox回归参数
        indices = jt.arange(N)
        selected_bbox_preds = bbox_preds[indices, labels]  # [N, 4]

        # 按照batch_offsets拆分成list返回
        cls_batch = []
        box_batch = []
        start_idx = 0
        for i in range(len(proposals)):
            end_idx = batch_offsets[i+1]
            cls_batch.append(cls_scores[start_idx:end_idx])
            box_batch.append(selected_bbox_preds[start_idx:end_idx])
            start_idx = end_idx

        return cls_batch, box_batch




    def init_weights(self):
        import math
        import jittor.init as init

        # fc1 和 fc2：全连接层 Xavier 初始化
        init.xavier_uniform_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)

        init.xavier_uniform_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

        # cls_score：Xavier 初始化，bias 用先验概率
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        init.xavier_uniform_(self.cls_score.weight)
        init.constant_(self.cls_score.bias, bias_value)

        # bbox_pred：Xavier 初始化也可（更常见），或保持为0也行
        init.xavier_uniform_(self.bbox_pred.weight)
        init.constant_(self.bbox_pred.bias, 0)


'''