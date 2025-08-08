from jittor import nn
import jittor as jt

def cross_entropy_no_reduce(inputs, targets):
    log_probs= nn.log_softmax(inputs, dim=1)         # [N, C]
    one_hot= jt.nn.one_hot(targets, inputs.shape[1]) # [N, C]
    return -jt.sum(one_hot * log_probs, dim=1)        # [N]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha= alpha
        self.gamma= gamma
        self.reduction= reduction
      

    def execute(self, inputs, targets):
        ce_loss= cross_entropy_no_reduce(inputs, targets)  # [N]
        pt= jt.exp(-ce_loss)
        focal_loss= self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction== 'mean':
            return focal_loss.mean()
        elif self.reduction== 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def RPN_loss(results, lambda_reg=1.0, total_samples=256, max_pos=128):
    #cls_loss_fn= FocalLoss(alpha=0.25, gamma=2.0)#nn.CrossEntropyLoss()
    cls_loss_fn= nn.CrossEntropyLoss()
    total_cls_loss= 0.0
    total_reg_loss= 0.0
    valid_image_count= 0

    for img_idx, (anchor, cls_p, box_p, label_p, shift_gt) in enumerate(results):
        valid_mask= (label_p >= 0)
        cls_p= cls_p[valid_mask]
        label_p= label_p[valid_mask]
        pos_mask= label_p== 1
        neg_mask= label_p== 0

        if cls_p.numel()== 0:
            continue  #跳过无有效样本的图像

        pos_inds= jt.nonzero(pos_mask).squeeze(-1)
        neg_inds= jt.nonzero(neg_mask).squeeze(-1)

        num_pos= min(max_pos, pos_inds.shape[0])
        num_neg= min(total_samples - num_pos, neg_inds.shape[0])

        if num_pos > 0:
            pos_sample_inds= pos_inds[jt.randperm(pos_inds.shape[0])[:num_pos]]
        else:
            pos_sample_inds= jt.array([], dtype=jt.int64)

        if num_neg > 0:
            neg_sample_inds= neg_inds[jt.randperm(neg_inds.shape[0])[:num_neg]]
        else:
            neg_sample_inds= jt.array([], dtype=jt.int64)

        sample_inds= jt.concat([pos_sample_inds, neg_sample_inds], dim=0)

        sample_preds= cls_p[sample_inds]
        sample_labels= label_p[sample_inds].long()

        if jt.isnan(sample_preds).any() or jt.isnan(sample_labels).any():  #及时发现loss爆炸
            print(f"分类loss输入含NaN")

        cls_loss= cls_loss_fn(sample_preds, sample_labels)
        reg_loss= jt.zeros(1)

        if num_pos > 0:
            reg_preds= box_p[pos_mask]
            reg_targets= shift_gt[pos_mask]
            num_reg_pos= min(reg_preds.shape[0], num_pos)
            reg_sample_inds= jt.randperm(reg_preds.shape[0])[:num_reg_pos]
            reg_sample_pred= reg_preds[reg_sample_inds]
            reg_sample_target= reg_targets[reg_sample_inds]
            if jt.isnan(reg_sample_pred).any() or jt.isnan(reg_sample_target).any():
                  print(f"回归loss输入含NaN")
            reg_loss= nn.smooth_l1_loss(reg_sample_pred, reg_sample_target, reduction="mean")

        total_cls_loss += cls_loss
        total_reg_loss += reg_loss
        valid_image_count += 1

       # print(f"正样本: {pos_inds.shape[0]}, 负样本: {neg_inds.shape[0]}")

    if valid_image_count== 0:
        return jt.zeros(1), jt.zeros(1), jt.zeros(1)

    avg_cls_loss= total_cls_loss / valid_image_count
    avg_cls_loss= jt.clamp(avg_cls_loss, max=2.0)

    avg_reg_loss= total_reg_loss / valid_image_count
    avg_reg_loss= jt.clamp(avg_reg_loss, max=3.0) #裁剪

    total_loss= avg_cls_loss + lambda_reg * avg_reg_loss
    
    return total_loss, avg_cls_loss, lambda_reg * avg_reg_loss


def RPN_loss_sample(results, lambda_reg=1.0, total_samples=64, max_pos=32):
    cls_loss_fn= nn.binary_cross_entropy_with_logits  # Jittor 自带
    total_cls_loss= 0.0
    total_reg_loss= 0.0
    valid_image_count= 0

    sample_indices_per_image= []       # 采样索引
    sampled_anchors_per_image= []      # 采样的 anchor
    sampled_preds_per_image= []        # 采样的预测偏移（用于回归）
    sampled_targets_per_image= []      # 对应 GT 偏移量
    sampled_labels_per_image= []
    for img_idx, (anchor, cls_p, box_p, label_p, shift_gt) in enumerate(results):
        valid_mask= (label_p >= 0)
        cls_p= cls_p[valid_mask]
        label_p= label_p[valid_mask]
        box_p= box_p[valid_mask]
        anchor= anchor[valid_mask]
        shift_gt= shift_gt[valid_mask]

        pos_mask= (label_p== 1)
        neg_mask= (label_p== 0)

        if cls_p.numel()== 0:
            sample_indices_per_image.append([])
            sampled_anchors_per_image.append(jt.empty((0, 4)))
            sampled_preds_per_image.append(jt.empty((0, 4)))
            sampled_targets_per_image.append(jt.empty((0, 4)))
            continue

        pos_inds= jt.nonzero(pos_mask).squeeze(-1)
        neg_inds= jt.nonzero(neg_mask).squeeze(-1)

        num_pos= min(max_pos, pos_inds.shape[0])
        num_neg= min(total_samples - num_pos, neg_inds.shape[0])
        #采样正负样本索引
        pos_sample_inds= (pos_inds[jt.randperm(pos_inds.shape[0])[:num_pos]] if num_pos > 0 else jt.array([], dtype=jt.int64))
        neg_sample_inds= (neg_inds[jt.randperm(neg_inds.shape[0])[:num_neg]] if num_neg > 0 else jt.array([], dtype=jt.int64))
        sample_inds= jt.concat([pos_sample_inds, neg_sample_inds], dim=0)
        sample_indices_per_image.append(sample_inds.tolist())



        #采样后的分类预测和标签
        sample_cls_preds= cls_p[sample_inds]
        sample_labels= label_p[sample_inds].long()

        #计算分类损失
       
        cls_loss= cls_loss_fn(sample_cls_preds.squeeze(-1), sample_labels.float())
       # cls_loss= cls_loss_fn(sample_cls_preds, sample_labels)
        reg_loss= jt.zeros(1)

        # 采样回归部分预测和目标：
        # 正样本回归损失计算，负样本偏移用预测值，不计算回归loss
        sampled_box_preds= box_p[sample_inds]
        sampled_shift_gts= shift_gt[sample_inds]
        sampled_anchors= anchor[sample_inds]

        if num_pos > 0:
            pos_in_sample= (sample_labels== 1)
            if pos_in_sample.sum() > 0:
               # std= jt.array([0.1, 0.1, 0.2, 0.2]).unsqueeze(0)
                #pred_box_norm= sampled_box_preds[pos_in_sample] / std 
                reg_loss= nn.smooth_l1_loss(sampled_box_preds[pos_in_sample], sampled_shift_gts[pos_in_sample], reduction="mean")

            else:
                reg_loss= jt.zeros(1)
        else:
            reg_loss= jt.zeros(1)

        sampled_preds_per_image.append(sampled_box_preds)
        sampled_targets_per_image.append(sampled_shift_gts)
        sampled_anchors_per_image.append(sampled_anchors)
        sampled_labels_per_image.append(sample_labels)

        total_cls_loss += cls_loss
        total_reg_loss += reg_loss
        valid_image_count += 1

    if valid_image_count== 0:
        return jt.zeros(1), jt.zeros(1), jt.zeros(1), [], [], [], []
    print(f'cls  box {total_cls_loss / valid_image_count,  total_reg_loss / valid_image_count}')
    avg_cls_loss= total_cls_loss / valid_image_count
    #avg_cls_loss= jt.clamp(avg_cls_loss, max_v=2.0)

    avg_reg_loss= total_reg_loss / valid_image_count
   # avg_reg_loss= jt.clamp(avg_reg_loss, max_v=3.0)

    total_loss= avg_cls_loss + lambda_reg * avg_reg_loss

    #sample_pred, sample_anchor= resample_rpn_output_for_roi_head(sampled_preds_per_image, sampled_anchors_per_image,sampled_labels_per_image)
    return total_loss, avg_cls_loss, avg_reg_loss#, sample_pred, sample_anchor

def resample_rpn_output_for_roi_head(
    sampled_preds_per_image,
    sampled_anchors_per_image,
    sampled_labels_per_image,
    total_samples=128,
    max_pos=64
):
    final_sampled_preds_per_image= []
    final_sampled_anchors_per_image= []

    for preds, anchors, labels in zip(sampled_preds_per_image, sampled_anchors_per_image, sampled_labels_per_image):
        if preds.shape[0]== 0:
            final_sampled_preds_per_image.append(jt.empty((0, 4)))
            final_sampled_anchors_per_image.append(jt.empty((0, 4)))
            continue

        pos_mask= (labels== 1)
        neg_mask= (labels== 0)

        pos_inds= jt.nonzero(pos_mask).squeeze(-1)
        neg_inds= jt.nonzero(neg_mask).squeeze(-1)

        num_pos= min(max_pos, pos_inds.shape[0])
        num_neg= min(total_samples - num_pos, neg_inds.shape[0])

        pos_sample_inds= (
            pos_inds[jt.randperm(pos_inds.shape[0])[:num_pos]]
            if num_pos > 0 else jt.array([], dtype=jt.int64)
        )
        neg_sample_inds= (
            neg_inds[jt.randperm(neg_inds.shape[0])[:num_neg]]
            if num_neg > 0 else jt.array([], dtype=jt.int64)
        )

        sample_inds= jt.concat([pos_sample_inds, neg_sample_inds], dim=0)

        final_sampled_preds_per_image.append(preds[sample_inds])
        final_sampled_anchors_per_image.append(anchors[sample_inds])

    return final_sampled_preds_per_image, final_sampled_anchors_per_image


'''
def RPN_loss(results, lambda_reg=10.0):
    total_cls_loss= 0
    total_reg_loss= 0
    total_imgs= len(results)

    for _, cls_p, box_p, label_p, shift_gt in results:
        # 1. 获取正负样本索引
        pos_inds= jt.nonzero(label_p== 1).squeeze(-1)
        neg_inds= jt.nonzero(label_p== 0).squeeze(-1)

        num_pos= min(64, pos_inds.shape[0])
        num_neg= 128 - num_pos

        if num_pos > 0:
            pos_inds= pos_inds[jt.randperm(pos_inds.shape[0])[:num_pos]]
        if num_neg > 0:
            neg_inds= neg_inds[jt.randperm(neg_inds.shape[0])[:num_neg]]
        sample_inds= jt.concat([pos_inds, neg_inds], dim=0)

        # 2. 分类 loss（取前景概率）
        sample_preds= cls_p[sample_inds][:, 1]  # [N] 取出前景概率
        sample_labels= label_p[sample_inds].float()

        eps= 1e-6
        sample_preds= jt.clamp(sample_preds, min_v=eps, max_v=1 - eps)
        cls_loss= -(sample_labels * jt.log(sample_preds) + (1 - sample_labels) * jt.log(1 - sample_preds))
        cls_loss= cls_loss.mean()

        # 3. 回归 loss（SmoothL1，仅正样本）
        if num_pos > 0:
            reg_loss= nn.smooth_l1_loss(box_p[pos_inds], shift_gt[pos_inds])
        else:
            reg_loss= jt.zeros(1)  # 或 zeros((1,), dtype=jt.float32)


        total_cls_loss += cls_loss
        total_reg_loss += reg_loss

    total_cls_loss /= total_imgs
    total_reg_loss /= total_imgs
    #print('rpn loss')
    return total_cls_loss + lambda_reg * total_reg_loss, total_cls_loss, total_reg_loss
'''
def DECT_loss(cls_list, box_list, gt_box_list, gt_label_list, lambda_dect=1.0, max_samples=128):
    """
    cls_list: List of [N, num_classes]
    box_list: List of [N, 4]
    gt_label_list: List of [N], int, 0负样本, >0正样本, -1忽略样本
    gt_box_list: List of [N, 4]
    """
    cls_losses= []
    box_losses= []

    for cls_score, box_pred, gt_boxes, gt_labels in zip(cls_list, box_list, gt_box_list, gt_label_list):
        if cls_score.numel()== 0:
            continue

        pos_indices= jt.where(gt_labels > 0)[0]
        neg_indices= jt.where(gt_labels== 0)[0]

        num_pos= min(pos_indices.shape[0], max_samples // 4)      # 正样本最多 1/4
        num_neg= min(neg_indices.shape[0], max_samples - num_pos) # 负样本补满总数

        if num_pos > 0:
            perm_pos= jt.randperm(pos_indices.shape[0])[:num_pos]
            sampled_pos= pos_indices[perm_pos]
        else:
            sampled_pos= jt.array([], dtype=jt.int64)

        if num_neg > 0:
            perm_neg= jt.randperm(neg_indices.shape[0])[:num_neg]
            sampled_neg= neg_indices[perm_neg]
        else:
            sampled_neg= jt.array([], dtype=jt.int64)

        sampled_indices= jt.concat([sampled_pos, sampled_neg], dim=0)

        sampled_cls_score= cls_score[sampled_indices]
        sampled_box_pred = box_pred[sampled_indices]
        sampled_labels   = gt_labels[sampled_indices]
        sampled_gt_boxes = gt_boxes[sampled_indices]

        # 分类 loss（忽略 -1）
        cls_loss= nn.cross_entropy_loss(sampled_cls_score, sampled_labels, ignore_index=-1)

        # 回归 loss（仅正样本）
        pos_mask= sampled_labels > 0
        if pos_mask.sum() > 0:
            box_loss= nn.smooth_l1_loss(sampled_box_pred[pos_mask], sampled_gt_boxes[pos_mask], reduction="mean")
        else:
            box_loss= jt.array(0.0)

        cls_losses.append(cls_loss)
        box_losses.append(box_loss)

    if len(cls_losses)== 0:
        return jt.array(0.0), jt.array(0.0), jt.array(0.0)

    total_cls_loss= sum(cls_losses) / len(cls_losses)
    total_cls_loss= jt.clamp(total_cls_loss, max_v=2.0)

    total_box_loss= sum(box_losses) / len(box_losses)
    total_box_loss= jt.clamp(total_box_loss, max_v=3.0)

    total_loss= total_cls_loss + lambda_dect * total_box_loss

    return total_loss, total_cls_loss, total_box_loss
