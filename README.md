# MTCNN-TensorFlow-2

## 简介

人脸检测完整复现，Python3.7+Tensorflow2
从模型的搭建到数据集的加载预处理，模型的训练，完整的实现了人脸框检测和5个关键点的标注
![img](results/test3_ed.png)

## 食用方法

+ 安装环境，python3.3以上应该都可以，tensorflow2必装，可以装gpu版；建议python3.7 + tensorflow2-gpu + win10

```cmd
pip install tensorflow-gpu
pip install opencv-python
pip install tqdm
```

+ 使用

`python -m mtcnn.detect [图片路径]`   可以包含多张图片，循环检测，输出并显示检测结果

`python camera_demo.py` 调用摄像头检测人脸

----------------------------------------------

## 总结
