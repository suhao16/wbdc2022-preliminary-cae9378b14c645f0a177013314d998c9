## [2022中国高校计算机大赛-微信大数据挑战赛](https://algo.weixin.qq.com/)

###环境依赖（包括python版本、CUDA版本、操作系统等环境信息)
#### code1
 - os:          Ubuntu 16.04 LTS (GNU/Linux 4.4.0-200-generic x86_64)
 - GPU :        NVIDIA® V100 Tensor Core * 2
 - CPU :        Intel(R) Xeon(R) Silver 4212 CPU @ 2.20GHz
 - mem :        503G
 - disk:        477G
 - nvidia-smi:  NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2  
 - nvcc -V:     Cuda compilation tools, release 9.0, V9.0.176
 - python==3.7.10
 - pytorch==1.10.1+cu102
 - transformers==4.17.0
 - 预训练显存占用60g左右，耗时接近7天多，当时权重取loss=0.31313,epoch33最好
 - 微调模型显存占用30g左右，非全量大约12小时，全量数据14小时都是epoch2最好

#### code2 
 - 镜像： PyTorch  1.7.0 Python  3.8 Cuda  11.0
- GPU： RTX 3090 * 1 显存:24GB
- CPU： 14核 Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
- 内存：45GB
 - transformers==4.17.0


###代码结构
```python
wbdc2022-preliminary-cae9378b14c645f0a177013314d998c9
│  inference.sh       # 运行推理
│  init.sh            # 环境初始化
│  README.md
│  requirements.txt   # python 包
│  train.sh           # 模型训练 
│
├─data
│  ├─annotations
│  └─zip_feats
└─src
    ├─code1
    │  │  category_id_map_.py                   # category_id 和一级、二级分类的映射
    │  │  config_f_pgd_.py                      # 配置文件
    │  │  fintue_data_480_tw_bu_fc_.py          # 数据预处理
    │  │  fintue_data_480_tw_bu_fc_quan_.py     # 全量数据预处理
    │  │  fintue_model_roberta_.py              # 训练用模型
    │  │  fintue_model_roberta_fusion_.py       # 融合，推理用模型
    │  │  inference_fusion_2ge_.py              # 模型融合与推理文件
    │  │  main_PGD_EMA_.py                      # 模型训练入口
    │  │  main_PGD_EMA_quan_.py                 # 全量数据训练模型入口
    │  │  util.py                               # util函数
    │  │
    │  ├─log_file   # 存放日志文件
    │  │
    │  ├─save       # 存放模型权重 
    │
    ├─code2 
    │  ├─config#训练参数
    │  ├─dataset
    │  │   │   category_id_map_.py              # category_id 和一级、二级分类的映射
    │  │   │   data_helper.py                   # 数据准备
    │  ├─models
    │  │   │   singlestream_model.py            # 训练模型
    │  ├─utils                   			    # 模型配置
    │  │   │   adv_train.py
    │  │   │   create_optimizer.py
    │  │   │   dice_loss.py
    │  │   │   focal_loss.py
    │  │   │   util.py
    │  │  config.py                             #训练参数设置
    │  │  evaluate.py
    │  │  inference.py                          # 模型融合与推理文件
    │  │  main.py                               # 模型训练入口
    ├─pre_train
    │  │   category_id_map.py                           # category_id 和一级、二级分类的映射
    │  │   config.py                                    # 预训练配置文件
    │  │   pretrain_data_helper60260160_tw_bu_fc.py     # 预训练数据预处理
    │  │   pretrain_main4.py                            # 预训练模型入口
    │  │   pretrain_roberta4_noleaky.py                 # 预训练模型
    │  │   util.py                                      # util函数
    │
    └─roberta       # 存放roberta模型权重以及配置文件
```
###运行流程说明
- cd到wbdc2022-preliminary-cae9378b14c645f0a177013314d998c9目录下后依次运行三个文件
- init.sh: 初始化环境
- train.sh: 模型训练
  - 在src/code1/save中首先生成预训练模型权重best_pretrain_roberta_base.bin
  - 然后训练生成12weight.npy，Best_EMA_PGD_480_bs32_lr5e-5_preepoch33.bin
  和Best_EMA_PGD_480_bs32_lr5e-5_preepoch33_quan.bin三个文件用于融合推理。
- inference.sh: 模型融合与预测，用上面生成的三个文件推理生成result.csv

###算法模型介绍
#### code1
- 模型主体：参考了[QQ浏览器2021AI算法大赛赛道一第1名方案](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st) embedding layer分别提取视觉和文本特征。
  其中文本包含title,asr,ocr,长度限制为480。然后直接连接视觉和文本，过BERT，最后通过简单的 MLP 结构去预测二级分类的 id
- 模型预训练：mlm,itm    
- trick：pgd,ema
- 模型融合:采用简单的全量数据和非全量数据的2个模型，取预测结果的平均。

#### code2
- title+osr+acr截断后拼接，增加长度（128）
- 郭大baseline，roberta-base，文本拼接（前64+后64），labelSmooth（0.1）
- 增加EMA 
- 增加FGM对抗
- 增加PGD对抗
- 全量训练

###模型初赛B榜离线/在线结果等内容,如使用了开源的预训练模型（例如 huggingface 上的模型），请在 README 文件中进行说明，并附上开源模型的链接。
- B榜离线结果 
  - code1 非全量0.673 (A榜线上66.8)和全量0.8967 (A榜线上67.0464)，两个平均融合得A榜线上67.7157
  - code2 全量训练 A榜线上66.9，更换随机数进行融合，融合9个A榜线上68.1，融合12个A榜线上68.4
  - 对code1中2个模型的推理与code2中模型9个模型的推理进行平均融合，提交得A榜线上68.9
- B榜在线结果  
  - 使用A榜68.9的2+9模型推理提交得，0.693599
