# Faster RCNN Jittor实现

这个项目代码实现参考了博客：https://blog.csdn.net/weixin_44791964/article/details/105739918 
以及Pytorch版对应源码：
https://github.com/bubbliiiing/faster-rcnn-pytorch
实验中根据VOC2012中的voc_total_coco.json，划分样本类别均衡8：2进行训练和验证
## Jittor版本文件目录
FasterRCNN/
├──dataset/              #Pascal VOC数据集接口
├──log/ #两个框架实现的训练记录
├── model/               #模型组成文件pascal_anno
├── pascal_anno/     #划分数据集方法
├── util/                     #实现工具
├── main.py  #训练文件
├── test.py  #测试文件
├── requirement.txt   # 环境需求
└── README.md

## 训练
- 数据集

数据集需要预先下载Pascal数据集，在/pascal_anno下的split_data.py中指定voc_total_coco.json和图片目录/VOCdevkit/VOC2012/JPEGImages，划分训练和验证数据，得到两个Pascal Voc标注文件进行数据加载
- 环境
```
conda create -n 环境名 python=3.10
conda activate 环境名
cd ./FasterRCNN
pip install -r requriements.txt

#安装JDet，利用到其中的ROI Align或者ROI Pooling
cd ../JDet
pip install -r requirements.txt
python setup.py develop
```
- 训练

在main.py中配置后直接运行

- 测试

加载训练好的模型以及数据集，利用test.py测试

## 训练流程

1. 创建模型，加载预训练的CNN BackBone后转为模型特征提取器
2. BackBone：输入图片，由Backbone初步提取图像特征，多尺度特征图宽高为原图像下采样16倍
3. RPN：
	- 按照初始设置，预定义三种尺度：128，256，512大小和三种比例2:1、1:1、1:2共九个基础anchor。将特征图网格映射回原图尺寸，然后在每个网格位置生成以网格为中心的9个anchor，得到图像上的anchor。
	- 将Backbone提供的图像特征由3×3的卷积层提取局部特征，然后利用两个1×1卷积层分别得到anchor对应的回归偏移量和二分类分数，形成初步Proposal。
	- 利用回归偏移量和对应的anchor，形成ROI。由二分类的前景分数排序后对ROI执行nms筛选得到Top-K个Proposal。
	- **Loss**:在初始图像铺满的anchor和GT进行IoU计算，以IoU阈值筛选正负样本共256个。由anchor的回归偏移量和二分类分数计算RPN的Loss。
4. 检测器：
	- **样本匹配及筛选**：利用筛选的Proposal，与GT框进行IoU计算，以IoU阈值0.5进行样本匹配，得到一批数量均匀的正样本和负样本。为了更好得训练模型，代码中会将GT也加入Proposal，得到高质量的样本。
	- 对Proposal指示的ROI区域进行ROI Pooling或ROI Align得到ROI特征，检测器利用两个全连接层得到所有类别的分数，以及相对于ROI的回归偏移（每一类都会生成一个）。
	- **Loss**：利用匹配阶段分配的标签取出对应的偏移量，作为改类别的回归的预测结果。由对应回归结果和分类分数计算检测器的Loss。
5. Loss：由RPN Loss和检测器 Loss 加和得到最终Loss。

## 两个版本实现的性能


## Jittor版本性能
自己构建的Jittor版本训练性能上升缓慢，因此进行了更多epoch的训练。原因还不清楚。但从Loss曲线观察是检测器回归问题。

- loss


- AP@50



## Pytorch版本性能
Pytorch版本性能上升较快。
- loss

- AP@50

- 最终性能

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.614
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.193
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
```
## 流程图
