import os
import matplotlib
import jittor as jt
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

from .util import cvtColor, resize_image, preprocess_input, get_new_img_size
from .eval import Eval
from .map import get_coco_map, get_map



class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.02, nms_iou=0.3, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir

        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.std    = jt.array([0.1, 0.1, 0.2, 0.2] * num_classes).reshape(1, -1) 

        self.bbox_util  = Eval(self.std, self.num_classes)

        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    #检测图片
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")

        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        #RGB图像的预测
        image       = cvtColor(image)
        
        
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])#给原图像缩放短边为600的大小上
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with jt.no_grad():
            images = jt.array(image_data)


            roi_cls_locs, roi_scores, rois, _ = self.net(images)

            #image_shape=shape[0]
            #height, width = image_shape[0], image_shape[1]
            #input_height, input_width = image_shape[2], image_shape[3]

            #利用classifier的预测结果对建议框进行解码，获得预测框
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
 
            #没有检测到物体 直接返回
            if len(results[0]) <= 0:
                return 
            top_label = np.array(results[0][:, 5], dtype = 'int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        top_100  = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes= top_boxes[top_100]
        top_conf= top_conf[top_100]
        top_label= top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score  = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
      #  print(results)
        return 
    
    def on_epoch_end(self, epoch,draw=True):
        if epoch % self.period == 0 and self.eval_flag:
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                image       = Image.open(line[0])
                #------------------------------#
                #   获得预测框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)
            if draw:

                with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                    f.write(str(temp_map))
                    f.write("\n")
                
                plt.figure()
                plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Map %s'%str(self.MINOVERLAP))
                plt.title('A Map Curve')
                plt.legend(loc="upper right")

                plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
                plt.cla()
                plt.close("all")

                print("Get map done.")
                shutil.rmtree(self.map_out_path)
