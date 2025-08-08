import datetime
import json
import os
import random
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import jittor as jt
from jittor import nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from utils.util import format_coco_stats,format_per_class_results,loc2bbox,cvtColor,resize_image,preprocess_input, get_new_img_size

# 解码预测
class Eval():
    def __init__(self, std=None, num_classes=21):


        self.num_classes= num_classes     
        if std is None:
            self.std = jt.array([0.1, 0.1, 0.2, 0.2] * num_classes).reshape(1, -1) 
        else:
            self.std= std
    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)
        box_mins    = box_yx - (box_hw / 2.) #左上
        box_maxes   = box_yx + (box_hw / 2.) #右下
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1) # x y x y
        boxes *= np.concatenate([image_shape, image_shape], axis=-1) # [x y x y] * [w h w h]
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []
        bs      = len(roi_cls_locs)
        #batch_size, num_rois, 4
        rois    = rois.view((bs, -1, 4))
        for i in range(bs):
            #训练利用了std 预测要乘上
            roi_cls_loc = roi_cls_locs[i] * self.std
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])


            #预测结果行调整获得预测框
            # num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            roi         = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox    = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox    = cls_bbox.view([-1, (self.num_classes), 4])
            #行归一化，调整到0-1之间 后面再回来
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

 
            roi_score   = roi_scores[i]
            prob        = nn.softmax(roi_score, dim=-1)

            results.append([])
     #   print(prob.shape)
            for c in range(1, self.num_classes): #从1类开始 循环遍历每个类的概率 ，看看是否大于阈值

                c_confs     = prob[:, c] # 所有框的分数
                c_confs_m   = c_confs > confidence  #大于阈值才保留

                if len(c_confs[c_confs_m]) > 0: # 该类别是否有预测框

                    boxes_to_process = cls_bbox[c_confs_m, c]  #坐标
                    confs_to_process = c_confs[c_confs_m]  #置信度

                    #jitor的nsm需要五维度
                    # 先把 confs_to_process 变成 [N, 1]
                    scores = confs_to_process.unsqueeze(1)  # [N, 1]

                    # 然后在维度1上拼接
                    dets = jt.concat([boxes_to_process, scores], dim=1)  # [N, 5]
                    keep = jt.nms(  #对同个类别的框执行nms
                        dets,
                        nms_iou
                    )         
                    good_boxes  = boxes_to_process[keep] #保留 nms后效果较好的 idx
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c-1)  * jt.ones((len(keep), 1))
                 #  最终保留keep个框 ，生成长度维keep的lable对齐，而label 和文件映射 要 即c = 文件中的lable 名称映射   json文件的gt 类别从1开始


                    c_pred= jt.concat((good_boxes, confs, labels), dim=1).numpy() #label、分数、框的位置堆叠 变成一条
                    # 逐个添加进result里
                    results[-1].extend(c_pred) 

        if len(results[-1]) > 0: #有预测结果
            results[-1] = np.array(results[-1])
            box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2] #转化为 x y h w
            results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape) #解码归一化坐标得到最终预测

      #  print(results)
        return results  # [x y h w score lable]
        


    def eval(self,model, dataloader, epoch,gt_json_path, save_path=None,log_path=None,save_json=None):
        model.eval()
        results = []
        total=len(dataloader.dataset)
        processed=0
        print("----------start evalution---------")
        with open(log_path, 'a') as f:
            f.write("----------start evalution---------\n")
        for i, (images, _, _, img_ids, shape) in enumerate(dataloader):

            box, scores, rois ,_ = model(images)  #outputs是列表，长度=batch_size
            image_shape=shape[0]
            height, width = image_shape[0], image_shape[1]
            input_height, input_width = image_shape[2], image_shape[3]
            outputs =self.forward(box, scores, rois[0],image_shape=(height, width),input_shape=(input_height, input_width))
            for b_idx, output in enumerate(outputs):
                image_id = int(img_ids[b_idx])
           # print(outputs)
            if len(output) == 0:
                continue
            else:
                for det in output:
                    x, y, w, h, score, label = det 
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score)
                    })

            processed += 1
            if processed % 50 == 0 or processed == total:
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                msg = f"[{now} eval] INFO: [{processed}/{total}] images processed"
                print(msg)
                if log_path is not None:
                    with open(log_path, 'a') as f:
                        f.write(msg + '\n')
       # print(results)     
        os.makedirs(save_path, exist_ok=True)
        save_json=os.path.join(save_path,"res.json")

        # === COCO 评估 ===
        # 加载原始标注 JSON 文件
        if save_json is not None:
            if save_json.endswith(".json"):
                with open(save_json, 'w') as f:
                    json.dump(results, f)
            else:
                raise ValueError(f"path must end with '.json',got: {save_json}")

        coco_gt = COCO(gt_json_path)
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
   


        #获取格式化文本
        global_str = format_coco_stats(coco_eval.stats)
        per_class_str = format_per_class_results(coco_eval, cat_id_to_name)


        #存模型
        map_50_95 = coco_eval.stats[0]  # AP@[IoU=0.5:0.95]
        model_filename = f"{epoch}_mAP{map_50_95:.4f}.pkl"
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, model_filename)
        jt.save(model.state_dict(), model_path)


        result_str = global_str + "\n" + per_class_str + "\n" + f"[Model Saved] {model_path}"
        print(result_str)

 
        with open(log_path, 'a') as f:#写入log
            f.write(result_str + "\n\n")







def get_random_colors(num_colors):
    base_colors = plt.cm.get_cmap('tab20').colors  # 20种颜色，范围是0~1的RGB
    colors = [tuple(int(c * 255) for c in color) for color in base_colors]
    random.shuffle(colors)
    return colors[:num_colors]



def detect_image(net, image_path, class_name, crop=False, count=False, nms_iou=0.3, confidence=0.8, num_classes=21):
    """
    图像检测推理并绘制结果

    参数：
    - net: Jittor检测模型
    - image_path: 待检测图片路径
    - class_name: 类别名称列表
    - crop: 是否裁剪目标并保存
    - count: 是否打印每类目标数量
    - nms_iou: NMS阈值
    - confidence: 置信度阈值
    - num_classes: 类别数

    返回：
    - PIL Image对象，绘制好检测框和标签
    """

    assert os.path.exists(image_path), "图像路径不存在"
    image = Image.open(image_path)

    bbox_util = Eval(num_classes)
    colors = get_random_colors(num_classes)

    # 原图尺寸(h, w)
    image_shape = np.array(image.size[::-1])
    # 计算resize后的尺寸，短边600
    input_shape = get_new_img_size(image_shape[0], image_shape[1])

    # 转成RGB防止灰度图报错
    image = cvtColor(image)
    # resize到input_shape大小
    image_data = resize_image(image, [input_shape[1], input_shape[0]])
    # 预处理并调整为(1, C, H, W)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    with jt.no_grad():
        images = jt.array(image_data)
        roi_cls_locs, roi_scores, rois, _ = net(images)

        # 预测后处理（解码+NMS）
        results = bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                                    nms_iou=nms_iou, confidence=confidence)

        # 无目标，直接返回原图
        if len(results[0]) <= 0:
            return image

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

    # 设置字体和框线粗细
    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=int(np.floor(3e-2 * image.size[1] + 0.5)))
    thickness = max(1, int((image.size[0] + image.size[1]) / np.mean(input_shape)))

    # 统计类别数量
    if count:
        print("top_label:", top_label)
        classes_nums = np.zeros([num_classes], dtype=int)
        for i in range(num_classes):
            num = np.sum(top_label == i)
            if num > 0:
                print(f"{class_name[i]} : {num}")
            classes_nums[i] = num
        print("classes_nums:", classes_nums)

    # 裁剪保存目标
    if crop:
        dir_save_path = "img_crop"
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        for i, c in enumerate(top_label):
            top, left, bottom, right = top_boxes[i]
            top = max(0, int(np.floor(top)))
            left = max(0, int(np.floor(left)))
            bottom = min(image.size[1], int(np.floor(bottom)))
            right = min(image.size[0], int(np.floor(right)))

            crop_image = image.crop([left, top, right, bottom])
            crop_image.save(os.path.join(dir_save_path, f"crop_{i}.png"), quality=95, subsampling=0)
            print(f"保存目标裁剪图 crop_{i}.png 至 {dir_save_path}")

    # 绘制检测框和标签
    for i, c in enumerate(top_label):
        predicted_class = class_name[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box
        top = max(0, int(np.floor(top)))
        left = max(0, int(np.floor(left)))
        bottom = min(image.size[1], int(np.floor(bottom)))
        right = min(image.size[0], int(np.floor(right)))

        label_text = f"{predicted_class} {score:.2f}"
        draw = ImageDraw.Draw(image)

        try:
            bbox = font.getbbox(label_text)
            label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        except AttributeError:
            label_size = font.getmask(label_text).size

        if top - label_size[1] >= 0:
            text_origin = (left, top - label_size[1])
        else:
            text_origin = (left, top + 1)

        # 画框，防止坐标错误
        for t in range(thickness):
            x0, y0 = left + t, top + t
            x1, y1 = right - t, bottom - t
            if x1 < x0 or y1 < y0:
                break
            draw.rectangle([x0, y0, x1, y1], outline=colors[c])

        # 标签背景框
        draw.rectangle([text_origin, (text_origin[0] + label_size[0], text_origin[1] + label_size[1])], fill=colors[c])
        # 标签文字
        draw.text(text_origin, label_text, fill=(0, 0, 0), font=font)

        del draw

    return image