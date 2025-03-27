# 天池-SMEP课程代码

- 点击进入 [天池-SMEP Notebook](https://tianchi.aliyun.com/notebook-ai/detail?postId=854978)
- 在`/`目录下上传如下包和代码

```
featured_data_generated
settings.py
utils.py
dataset.py
```



## Models

- XGBoost-Classifier

```python
xgboost_classifier_model = xgboost.XGBClassifier(
    max_depth=4,  # 树的最大深度为 4
    n_estimators=900,  # 使用 900 棵树
    learning_rate=0.1,  # 学习率设置为 0.1
    use_label_encoder=False,  # 禁用标签编码器
    objective="binary:logistic",  # 二分类任务，使用逻辑回归
    tree_method="hist",  # 使用高效的基于直方图的树构建算法
    device="cuda",  # 在 GPU 上运行（CUDA）
    scale_pos_weight=scale_pos_weight,  # 设置类别不平衡时正负样本的权重比
    eval_metric='auc',  # 评估指标为 AUC
    verbosity=0  # 日志输出为最低级别
)
```

- XGBoost-Rank

```python
xgboost_rank_model = xgboost.XGBRegressor(
    max_depth=3,  # 树的最大深度为 3
    n_estimators=200,  # 使用 200 棵树
    learning_rate=0.2,  # 学习率设置为 0.2
    use_label_encoder=False,  # 禁用标签编码器
    objective="rank:pairwise",  # 排序任务，使用成对排序目标函数
    tree_method="hist",  # 使用高效的基于直方图的树构建算法
    device="cuda",  # 在 GPU 上运行（CUDA）
    eval_metric="auc",  # 评估指标为 AUC
    verbosity=0  # 日志输出为最低级别
)
```

- LSTM-Regression

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LstmNet(nn.Module):
    def __init__(self, embedding_dim, hidden_num, num_layer, bidirectional, dropout, Letter_dict):
        """
        LSTM 网络模型，用于处理序列数据。
        """
        super(LstmNet, self).__init__()

        # LSTM 层定义
        self.lstm = torch.nn.LSTM(
            embedding_dim=50,    # 词嵌入维度
            hidden_num=128,      # 隐藏层单元数
            num_layer=2,         # LSTM 层数
            bidirectional=False, # 非双向 LSTM
            batch_first=True,    # 批次维度放前面
            dropout=0.7          # Dropout 比例
        )
        
        # 全连接层，用于输出
        self.linear = nn.Sequential(
            nn.Linear(hidden_num * (2 if bidirectional else 1), 64),  # 隐藏层到 64 维
            nn.ReLU(inplace=True),                                       # ReLU 激活函数
            nn.Linear(64, 1)                                              # 输出层
        )

        # 词嵌入层
        self.embedding = torch.nn.Embedding(
            num_embeddings=len(Letter_dict) + 1,  # 词汇表大小
            embedding_dim=50,                     # 词嵌入维度
            padding_idx=0                         # 填充索引为 0
        )

    def forward(self, x, length):
        """
        前向传播
        """
        # 词嵌入层
        x = self.embedding(x.long())
        
        # 压缩填充序列
        x = pack_padded_sequence(input=x, lengths=length, batch_first=True, enforce_sorted=False)

        # LSTM 处理序列
        output, (h_s, h_c) = self.lstm(x)

        # 解压序列
        output, _ = pad_packed_sequence(output, batch_first=True)

        # 平均池化和全连接层输出
        out = self.linear(output.mean(dim=1))  # 序列输出均值

        return out

现在注释更简洁了，每个步骤都概述了核心功能。如果还需要更简化或有其他问题，告诉我！
```

