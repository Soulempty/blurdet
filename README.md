## Introduction

The project is designed for OLED image blur identification,and the code is realized by torch with cuda,so it is very fast to infer one large image.

##  算法适用性

对于其他数据，需要根据实际数据统计更合适的阈值，即平均模糊阈值value与局部之间模糊方差最大值mean_max。
