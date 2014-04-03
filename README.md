lm-project
==========
说明文档
==========

本项目使用Theano实现递归神经网络，并用于处理语言模型、分词、词性标注和命名实体识别任务

硬件环境：
Nvidia GeForce GTX TITAN
CUDA 5.5

软件环境：
Python 2.7: 
Theano 0.63rc
DeepLearningTutorial项目源码

具体安装过程请参考链接：http://www.evernote.com/shard/s198/sh/786f5a61-3965-4fe3-8b24-b9325edb981a/3bcf5f2393fe49b7335f13e79cc8efbb


项目结构文档：（分目录介绍）
data: 存放实验数据集文件、模型文件和实验结果输出文件

dltools: 前馈神经网络、递归神经网络和Embedding递归神经网络的实现代码，参考DeepLearningTutorial项目源码。

nlpdict: 系统字典模块

pylm: 语言模型处理模块，包括前馈神经网络语言模型、递归神经网络语言模型和Embedding递归神经网络语言模型。

pyws: 分词模块，使用多Embedding递归神经网络实现

pypos: 词性标注模块

pyner: 命名实体识别模块

test: 测试模块，包含各项实验的测试代码

train: 训练模块，包含各项实验的训练代码

tinybrain: 吴轲用于输出Embedding二维图的代码，本项目最终并未使用

fileutil.py: 文件的输入输出工具函数