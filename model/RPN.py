import math
from jittor import nn
import jittor as jt
import numpy as np


import jittor.init as init
from utils.util import loc2bbox

class ProposalCreator():
    def __init__(
        self, 
        nms_iou             = 0.7,
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 600,
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 300,
        min_size            = 16
    ):

        self.nms_iou            = nms_iou

        #训练用到的
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms

        # 预测用到的
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms
        self.min_size           = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.,mode="train"):
        if mode == "train": #训练还是预测处理
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else:
            n_pre_nms   = self.n_test_pre_nms
            n_post_nms  = self.n_test_post_nms


        anchor = jt.array(anchor).type_as(loc)
    #    print(f"anchor {anchor.shape}")

        roi = loc2bbox(anchor, loc)  #转化为proposal xyxy

        #防止建议框超出图像边缘 要clamp
        roi[:,[0,2]] = jt.clamp(roi[:, [0,2]], min_v = 0, max_v = img_size[1])
        roi[:,[1,3]] = jt.clamp(roi[:, [1,3]], min_v = 0, max_v = img_size[0])
     #   print(roi.shape)

        min_size= self.min_size * scale #x2-x1    y2-y1
        keep  = jt.where(((roi[:,2] - roi[:,0]) >= min_size) & ((roi[:,3] - roi[:,1]) >= min_size))[0]


        roi = roi[keep, :]
        score = score[keep]

        # 根据得分进行排序，选取top 个 nms
        order,_= jt.argsort(score, descending=True)
        order = jt.array(order)
        if n_pre_nms > 0:
            order   = order[:n_pre_nms]
        roi= roi[order, :]
        score= score[order]

        #nms 因为jittro 输入是 x y x y score要先cat
        score_2d = score.unsqueeze(1)
        det = jt.concat([roi, score_2d], dim=1)
        keep    = jt.nms(det, self.nms_iou)

       #这里如果nms后数量不够需要 复制现有的 ，可以不复制但后续样本可能减少
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep        = jt.concat([keep, keep[index_extra]])
        keep    = keep[:n_post_nms]
        roi     = roi[keep]

        return roi

# 生成基础的achor
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

#进行拓展对应到所有特征点上
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):

    #计算网格中心点
    shift_x             = np.arange(0, width * feat_stride, feat_stride)
    shift_y             = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y)
    shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    A       = anchor_base.shape[0]
    K       = shift.shape[0]
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    #---------------------------------#
    #   所有的先验框
    #---------------------------------#
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

class RPN(nn.Module):
    def __init__(self,in_channel,
                 out_channel,
        ratios = [0.5, 1, 2],
        anchor_scales= [8, 16, 32], 
        feat_stride= 16,
        n_train_pre_nms=6000, #训练时 nms 前保留多少 proposal
        n_train_post_nms=1000, # 训练时 nms后保留多少
        nms_iou=0.7, # nms处理的iou
        n_test_pre_nms=3000, # 预测时nms前保留多少
        n_test_post_nms=600):
        '''
        in_channel: 输入特征图维度,
        out_channel: 输出行进RP操作的维数,
        '''
        super(RPN).__init__()
        # 9 4
        #生成anchor
        self.anchor_base    = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor            = self.anchor_base.shape[0]

      
        #滑动窗口 输入通道×每个通道一个卷积核×相加融合，得到一个输出通道：重复输出通道次
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=(3,3),stride=1,padding=1)

      
        #四维偏移坐标, 输出anchor的四倍 每四个为一个anchor的  1x1conv
        self.loc=nn.Conv2d(out_channel,4*n_anchor ,kernel_size=(1,1),stride=1)

        #二分类分数 1x1conv -> softmax 得到二分类类别
        self.score=nn.Conv2d(out_channel,2*n_anchor ,kernel_size=(1,1),stride=1)

        self.feat_stride    = feat_stride
        self.proposal = ProposalCreator(
                        n_train_pre_nms=n_train_pre_nms, 
                        n_train_post_nms=n_train_post_nms, 
                        nms_iou=nms_iou, 
                        n_test_pre_nms=n_test_pre_nms, 
                        n_test_post_nms=n_test_post_nms)
        self.init_weights()


    def execute(self, x,img_size, scale=1.):
        B, C, H, W = x.shape
        # 滑动窗口
        feature = nn.relu(self.conv1(x))

        #框
        pred_box = self.loc(feature)     # [B, 4*K, H, W]
        pred_box =pred_box.permute(0,2,3,1).contiguous().view(B,-1, 4) # B, H W K ,4
        #分数

        pred_cls = self.score(feature)      # [B, 2*K, H, W]
        pred_cls =pred_cls.permute(0,2,3,1).contiguous().view(B,-1, 2) # B, H W K ,2

        #softmax 
        rpn_softmax_score = nn.softmax(pred_cls,dim=-1)
        fg_score = rpn_softmax_score[:,:,1].contiguous()
        fg_score=fg_score.view(B,-1)  #用softmax得到前景分数 B ，H W K ，1 用来后面操作
        
        #铺满
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, H, W)

        #解码生成候选区,这里为了匹配roi align 需要一个idx 有可能不是整齐的
        rois=[]
        roi_idx=[]

        mode = "train" if self.is_training() else "eval"
        for i in range(B):
            roi  = self.proposal(pred_box[i], fg_score[i], anchor, img_size, scale = 1,mode=mode)
            batch_index = i * jt.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_idx.append(batch_index.unsqueeze(0))
          

        rois = jt.concat(rois, dim=0)         # [sum(M_i), 4]
        roi_idx = jt.concat(roi_idx, 0)   # [sum(M_i),] 记录 roi 属于哪张图
        anchor =jt.array(anchor).unsqueeze(0)
      #  print(roi_idx[2990:3020])
      #  print(f"rpn pred_box {pred_box.shape}")
        return pred_box,pred_cls,rois,roi_idx,anchor # roi [2,3000,4]  idx [2,3000,1]


            






    #根据 特征图 HW 和采样倍率生成在原图像上的 anchor
    def _generate_anchor(self,W,H,stride=16):
        scales = self.setting['scale']
        aspects = self.setting['aspect']
        K = len(self.setting['scale']) * len(self.setting['aspect'])
        anchor_sizes = []
        for scale in scales:
            for ratio in aspects:
                w = scale * math.sqrt(ratio)
                h = scale / math.sqrt(ratio)
                anchor_sizes.append([w, h])  #得到不同比例anchor 
        anchor_sizes = jt.array(anchor_sizes)  # [A, 2]
        
    
        
        #生成所有特征图位置中心点 (x, y)
        shift_x = jt.arange(W).float() * stride + stride / 2  #映射回原图 对应坐标
        shift_y = jt.arange(H).float() * stride + stride / 2
        x, y = jt.meshgrid(shift_x, shift_y)# [H,W]
        xy = jt.stack([x, y], dim=-1).reshape(-1, 2)  # [H*W, 2]   
        L = xy.shape[0]

        # 对每个中心点+每个 anchor 尺寸，生成 [x1,y1,x2,y2]
        # xy: [L, 2] → [L, 1, 2]
        # anchor_sizes: [K, 2] → [1, K, 2]
        centers = xy.reshape(L, 1, 2)
        sizes = anchor_sizes.reshape(1, K, 2)

        x1y1 = centers - sizes / 2  #左上角
        x2y2 = centers + sizes / 2  #右下角

        #生成 anchor: [H*W*A, 4]   原始特征图绝对坐标
        anchor = jt.concat([x1y1, x2y2], dim=-1).reshape(-1, 4)  
        return anchor
    

    #初始化
    def init_weights(self):
        import jittor.init as init
        #滑窗
        init.xavier_uniform_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0)
        #坐标回归
        init.xavier_uniform_(self.loc.weight)
        init.constant_(self.loc.bias, 0)
        #分类分数
        init.xavier_uniform_(self.score.weight)
        init.constant_(self.score.bias, 0)  
