<h2 align = "center">meanshift 算法在图像中的应用</h2>

#### 主要应用

- 保持边缘的平滑滤波 （√，已完成）
- 图像分割 （√，已完成）
- 目标跟踪（×，暂时不在我的研究范围内）

------

#### 基本原理

- 论文：
  - 《Mean shift: A robust approach toward feature space analysis》
  - 《An Implementation of the Mean Shift Algorithm》

- 博客：
  - [mean shift 图像分割（一、二、三）](https://blog.csdn.net/u011511601/article/details/72843247) 
  - [均值偏移（ mean shift ）？](https://www.zhihu.com/question/67943169) 

------

#### 实验结果

-  `mandrill` （左上角为原图）

| ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/original.jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(8%2C%208).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(8%2C%2016).jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(16%2C%204).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(16%2C%208).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(16%2C%2016).jpg) |
| ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(32%2C%204).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(32%2C%208).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/mandrill_results/(32%2C%2016).jpg) |

-  `others` 

| 图片名称  |                           原始图像                           |                      (hs, hr) = (4, 8)                       |                      (hs, hr) = (8, 16)                      |                           分割结果                           |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   boat    | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/boat_results/original.jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/boat_results/(4%2C%208).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/boat_results/(8%2C%2016).jpg) | <img src="https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/boat_results/seg(8%2C%2016).png" style="zoom: 67%;" /> |
| cameraman | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/cameraman_results/original.jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/cameraman_results/(4%2C%208).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/cameraman_results/(8%2C%2016).jpg) | <img src="https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/cameraman_results/seg(8%2C%2016).png" style="zoom: 67%;" /> |
|   house   | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/house_results/original.jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/house_results/(4%2C%208).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/house_results/(8%2C%2016).jpg) | <img src="https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/house_results/seg(8%2C%2016).png" style="zoom: 67%;" /> |
|  peppers  | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/peppers_results/original.jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/peppers_results/(4%2C%208).jpg) | ![](https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/peppers_results/(8%2C%2016).jpg) | <img src="https://raw.githubusercontent.com/chubbylhao/meanshift/main/image/results/peppers_results/seg(8%2C%2016).png" style="zoom: 67%;" /> |

------

#### 最后

代码还有许多地方可以改进（效率比较低），但若仅仅是出于对原理的理解是足够了

另外吐槽一句，python跑循环是真的慢，做实验的时候都不敢用大图~~