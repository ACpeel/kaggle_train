# Kaggle 训练指北

> 旨在通过Kaggle的免费GPU算力，进行模型训练



### 前提

1. 科学上网

> 基本操作，科研不二之选

2. 所需数据集网址，可通过[谷歌数据集搜索](https://datasetsearch.research.google.com/)

> 如果查询到了kaggle上已开源的最好，也可以通过下载链接的方式
>
> 大多数据集下载连接依靠[谷歌云盘](https://drive.google.com/drive/home)寄存数据集

3. google的账户

>  可以不用，但最好有，方便进行一键注册等

4. 准备好自己的github仓库（SSH）

> 一些私有库（private repo）可能需要授权，本教程通过ssh密钥授权，可能对你的账户或库造成一些影响，如有介意，请调整权限设置



### 指北

> 准备好上述内容后就可以开始食用

##### Kaggle平台准备

- 注册kaggle账号

[Kaggle](https://www.kaggle.com/account/login)

可通过谷歌账户一键注册登陆，或者使用email注册

- 验证手机号

完成登陆后，找到[个人设置页面](https://www.kaggle.com/settings)  —> Phone verification

点击verify进行验证（支持+86中国号码，或者通过sms-activate等虚拟号平台（自行尝试，未经验证可行性））

验证过程可能需要手机扫码进行人脸验证，照做即可

- 创建[notebook](https://docs.jupyter.org.cn/en/latest/#google_vignette) 

> Notebook是一种可共享的文档，它结合了计算机代码、纯语言描述、数据、丰富的可视化效果（如 3D 模型、图表、图形和图片）以及交互式控件。笔记本与编辑器（如 JupyterLab）一起，为代码原型设计和解释、数据探索和可视化以及与他人分享想法提供了一个快速的交互式环境。

点击kaggle默认页面左上角的Create创建一个notebook

![Notebook_Init](./assets/notebook_init.png)

点击上方settings—> Accelerator —> GPU T4 * 2，选择一个你喜欢的GPU/TPU

> 弹出提示 每周有 30 小时开机额度

![T4](./assets/GPU_2_T4.png)

Turn on以后点击右上角开机，稍作等待即可开始操作这台GPU云服务器

![start](./assets/start.png)



<img src="./assets/gpu_info.png" alt="gpu_info" style="zoom:150%;" />

> 欣赏一下免费的30GB  GPU ， 30GB RAM，50GB+ 存储空间，即可**关机**，进行其他准备

##### 库准备

> 这里以本仓库为例

##### 训练演示（2x T4 / 30GB RAM）

- 在 Kaggle Notebook 右侧 `Add data` 添加数据集 `phucthaiv02/butterfly-image-classification`（会挂载到 `/kaggle/input/butterfly-image-classification`）
- 克隆仓库后，在仓库根目录执行：
  - 下载到工作目录（可选）：`!make -C kaggle_train dataset`
  - 使用业界强基线（默认 ConvNeXtV2，自动用 2 张 T4）：`!make -C kaggle_train train`

常用配置（推荐先从这些开始）：
- EfficientNetV2：`!python kaggle_train/scripts/train.py --model tf_efficientnetv2_m --batch-size 64 --epochs 20 --num-workers 8`
- 更省显存：`!python kaggle_train/scripts/train.py --model tf_efficientnetv2_s --batch-size 96 --epochs 20 --num-workers 8`
- 关闭多卡（调试用）：`!python kaggle_train/scripts/train.py --no-multi-gpu`

训练输出：
- 最佳权重：`kaggle_train/outputs/butterfly_run/best.pt`

