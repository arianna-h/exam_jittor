from jittor import nn
from model.RPN import RPN
from model.Fast_RCNN import Fast_RCNN

from model.r50 import resnet50
from model.vgg import decom_vgg16

class Faster_RCNN(nn.Module):
    def __init__(self,num_classes,  
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False,
                    n_train_pre_nms=6000, #训练时 nms 前保留多少 proposal
                    n_train_post_nms=1000, # 训练时 nms后保留多少
                    nms_iou=0.7, # nms处理的iou
                    n_test_pre_nms=3000, # 预测时nms前保留多少
                    n_test_post_nms=600 ):
        super().__init__()
        self.feat_stride = feat_stride
        if backbone=='vgg':
            extractor, classifier = decom_vgg16(pretrained)
            self.extractor=extractor
            self.rpn=RPN(in_channel=512,out_channel=512,
                        ratios= ratios, 
                        anchor_scales= anchor_scales,
                        feat_stride= self.feat_stride,
                        n_train_pre_nms=n_train_pre_nms, 
                        n_train_post_nms=n_train_post_nms, 
                        nms_iou=nms_iou, 
                        n_test_pre_nms=n_test_pre_nms, 
                        n_test_post_nms=n_test_post_nms
                )
            self.head=Fast_RCNN(n_class=num_classes,in_channel=4096,roi_size=(7,7),spatial_scale=1,classifier=classifier,backbone=backbone)

        elif backbone=='resnet':
            extractor, classifier = resnet50(pretrained)
            self.extractor=extractor
            self.rpn=RPN(in_channel=1024,out_channel=512,
                        anchor_scales = anchor_scales,
                        feat_stride = self.feat_stride,
                        n_train_pre_nms=n_train_pre_nms, 
                        n_train_post_nms=n_train_post_nms, 
                        nms_iou=nms_iou, 
                        n_test_pre_nms=n_test_pre_nms, 
                        n_test_post_nms=n_test_post_nms
                )
            self.head=Fast_RCNN(n_class=num_classes,in_channel=2048,roi_size=(14,14),spatial_scale=16,classifier=classifier,backbone=backbone)


    #对齐pytorch版本 ，这里也分为不同模式
    def execute(self, x, scale=1., mode="forward"): #直接预测 用于eval
        if mode == "forward":
            #输入原始大小
            img_size        = x.shape[2:] 

            base_feature    = self.extractor.execute(x)

            _, _, rois, roi_indices, _  = self.rpn.execute(base_feature,img_size, scale )

            roi_cls_locs, roi_scores    = self.head.execute(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor": #网络提取特征


            base_feature    = self.extractor.execute(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            # proposal
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.execute(base_feature,img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #print(roi_indices)
            #传入大的
            #分类结果和回归结果
            roi_cls_locs, roi_scores    = self.head.execute(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


   
    '''
    def execute(self, x, gt_box_list=None, gt_label_list=None):
    
        
        feature = self.backbone(x)

        cls = []
        box = []
        #按batch处理
        proposal, rpn_loss, rpn_cls_loss, rpn_box_loss = self.rpn(feature, gt_box_list)  # 假设返回 [(anchor, cls_b, box_b, label_b, target_b), ...]

        #print(f'处理的batch {len(proposal)}')

        for i in range(len(proposal)):
            anchor,box_p = proposal[i]  # N, 4 等

            # 利用偏移量还原提取roi feature
            cls_b, box_b = self.detector(feature[i], anchor, box_p)
            #print(f'检测器检测成功 类别:{cls_b.shape} 框:{box_b.shape}')
            cls.append(cls_b)
            box.append(box_b)

        #训练推理判断
        if not self.is_training():
            box_out, score_out, label_out = self.post_process(box, cls)
            return box_out, score_out, label_out
        else:
            

            matched_label_list = []
            matched_gtbox_list = []

            #这里用proposal的anchor作为proposals传入匹配函数
            for i in range(len(proposal)):
                anchor,_ = proposal[i]
                pred_box=box[i]
                matched_labels, matched_gt_boxes = match_proposals_to_gt(pred_box,anchor,gt_box_list[i], gt_label_list[i])
                matched_label_list.append(matched_labels)
                matched_gtbox_list.append(matched_gt_boxes)

            dect_loss, dect_cls_loss, dect_box_loss = DECT_loss(cls, box, matched_gtbox_list, matched_label_list)

            total_loss = rpn_loss + dect_loss
            return total_loss, [dect_loss, dect_cls_loss, dect_box_loss], [rpn_loss, rpn_cls_loss, rpn_box_loss]
    '''
