## [2022中国高校计算机大赛-微信大数据挑战赛](https://algo.weixin.qq.com/)

大赛官方网站：https://algo.weixin.qq.com/

### 数据介绍
详见 [data/README.md](data/README.md)，请确保先检查数据下载是否有缺漏错误。

### Results
1. 官方baseline `save/v`：0.5802
2. title+osr+acr截断后拼接 `save/v2`：0.6217
3. title+osr+acr截断后拼接，增加长度（128） `save/v4`：0.6254
4. 郭大baseline，mac-bert，文本拼接长度（128） `save/v7`：0.6332
5. 郭大baseline，roberta-base，文本拼接长度（128） `save/v8`：（线下0.6511）0.6332
6. 同[5]，文本拼接（前64+后64），labelSmooth（0.1） `save/v9`：0.6369
7. 同[6]，增加EMA `save/v10`：0.6605
8. 同[7]，增加FocalLoss，但效果较差
9. **同[7]，增加FGM对抗 `save/v12`：0.6683**
10. 同[7]，线性层增加Tanh激活函数 `save/v13`：效果较差
11. 同[7]，增加PGD对抗 `save/v14`：0.6643
12. 同[9]，全量训练 `save/v15`：

### Completed
- EMA
- label smooth
- FGM对抗
- PGD对抗

### Todo
- roberta-base -> roberta-large
- TDIDF增加文本特征
- K折交叉
- Trick
  - 对抗训练（FGM、PGD、AWP）：最简单的是 FGM 和 PGD
  - F1 指标优化：通过对当前模型在验证集上的最优验证结果进行一个阈值搜索，得到一个可以使得 F1 值能够提升的最佳阈值
  - swa（模型权重平均）：DEEPNER，通过将模型训练最后几轮的权重进行平均
  - Rdrop：把对应的 kl loss 放缩到 0.x 的量级即可
- 双流模型
- 损失函数
  - focal loss
  - dice loss