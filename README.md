# SMEP

🔗 This repository code is derived from [the supplementary material](https://static-content.springer.com/esm/art%3A10.1038%2Fs41551-022-00991-2/MediaObjects/41551_2022_991_MOESM5_ESM.zip) in [the SMEP article](https://www.nature.com/articles/s41551-022-00991-2)

## Requirements

1. numpy==1.19.4  
1. pandas==0.25.3  
1. python==3.6.12  
1. scikit-learn==0.23.2  
1. torch==1.7.0+cu101  
1. torchvision==0.8.0+cu101  
1. xgboost==1.3.0  
1. tqdm==4.54.1  

## Installation

```python
pip install -r requirements.txt
```

## Data preparation

### Generate training and testing files

```
python generate_sample.py
```

### Generate sequences for searching  

> Use ```sequence_generated.py``` in ```./sequence_generated``` to generate the sequence for customized searching space, we offered sequences for peptides which length is 6 and the script to generate peptide sequences of length 7 in folder ```./sequence_generated```.

```bash
cd sequence_generated && mkdir 7_peptide_result
python3 sequence_generated.py
```

### Generate structual data for sequences  

> Use ```cal_pep_des.py``` in ```./featured_data_generated``` to generate structual data for Classification and Ranking stage from the sequences derived in the last step.

```bash
cd sample
python3 ../featured_data_generated/cal_pep_des.py
```

## Model Training

### Pipeline training

> Use ```train.py``` to get all the params for the three models(Classifcation, Ranking, Regressing). You can use customized training data or data generated from Grampa dataset.

### Incremental learning

> Use ```lstm_fine_tune.py``` for incremental learning. The augmented data was provided in folder ```./data/origin_data```. Using customized data validated in other wet-lab settings is optional.

## Searching for antimicrobial sequences

> Use ```predict.py``` to get the final searching result. For a vast searching space, you may use 'chunk' mechanism to avoid RAM shortage.



# WLab-Workflow

- Download resource

```bash
git clone https://github.com/BioGavin/SMEP.git
cd SMEP
```

- Create conda env

```bash
conda create -n smep python=3.6.12
conda activate smep
pip install -r requirements.txt
```

- Generate training and testing files

```bash
mkdir sample
python3 generate_dataset.py
```

- Train classification model

```bash
mkdir params
mkdir -p results/xgb_classifier_result
python3 train.py -md xgb_classifier --train_xgb_file sample/xgb_train_sample.csv --test_xgb_file sample/xgb_train_sample.csv > results/xgb_classifier_result/xgb_classifier_train.log
```

- Train ranking model

```bash
mkdir -p results/xgb_rank_result
python3 train.py -md xgb_rank --train_xgb_file sample/xgb_train_sample.csv --test_xgb_file sample/xgb_test_sample.csv > results/xgb_rank_result/xgb_rank_train.log
```

- Train regression model

```bash
mkdir -p results/lstm_result/regress
python3 train.py -md lstm --train_lstm_file sample/lstm_train_sample.csv --test_lstm_file sample/lstm_test_sample.csv > results/lstm_result/lstm_result_train.log
```

- Incremental learning

```bash
mkdir -p results/incremental_learning
python3 lstm_fine_tune.py --lstm_param_path params/regress_allmse_xx.pth --train_file_path data/origin_data/new_data_67.csv --save_parm_path params/finetune.pth > results/incremental_learning/incremental_learning.log
```

- Predict

```bash
# Generate sequences for searching
cd sequence_generated && mkdir 7_peptide_result
python3 sequence_generated.py

# Generate structual data for sequences
python3 ../featured_data_generated/cal_pep_des.py 7_peptide_result/7_peptide_rule_0.txt 7_peptide_result/7_peptide_rule_0.csv

# predict
mkdir prediction_results
python3 ../predict.py --lstm_param_path ../params/finetune.pth --result_save_path prediction_results --train_xgb_file ../sample/xgb_train_sample.csv --test_xgb_file ../sample/xgb_test_sample.csv --predict_xgb_classifier_file 7_peptide_result/7_peptide_rule_0.csv --save_xgb_classify_result True > prediction_results/predict.log
```

In `prediction_results` folder:

`classifier_feature_data.csv`: Positive result data predicted by XGBoost classification model.

`xgboost_classify.txt`: Positive sequences predicted by XGBoost classification model.

`top_500.csv`: The top 500 sequences predicted by the XGBoost sorting model. The MIC value is obtained by taking log10.

`lstm_result.csv`: MIC results predicted by LSTM model.



- Predict followed by cAMP

```bash
python3 gen_featured_data_for_cAMP.py task/camp.pos.tsv task/camp.pos.filtered.tsv task/camp.pos.filtered_featured_data.csv
mkdir -p task/prediction_results
python3 predict.py --lstm_param_path params/finetune.pth --result_save_path task/prediction_results --train_xgb_file sample/xgb_train_sample.csv --test_xgb_file sample/xgb_test_sample.csv --predict_xgb_classifier_file task/camp.pos.filtered_featured_data.csv --save_xgb_classify_result True > task/prediction_results/predict.log
```



# Note

## Datasets

- Positive dataset: Grampa数据集 (`data/origin_data/grampa.csv`) 包含了来自多个开放源数据集（如APD、DADP、DBAASP、DRAMP和YADAMP）的共计51,345个肽段实验结果。在数据预处理过程中，只选择了标准化的C-末端酰胺氨基酸序列，并使用Python数据分析库Pandas和Numpy工具包去除了长度超过50或少于5的异常值。此外，只保留了标记为对*S. aureus*具有特定活性的数据。对于同一抗菌肽标记的两个或更多冲突的最小抑菌浓度（MIC）值，采用了简单的几何平均策略得到最终的细菌-抗菌肽测量值。最终，Grampa数据集的预处理得到了1,762个阳性样本 (`data/filtered_data/positive.csv`) 
- Negative dataset: 5898个没有抗菌活性的负样本 (`data/filtered_data/negative.csv`) 是从UniProt数据库 (http://www.uniprot.org/)获取的 (`data/origin_data/origin_negative.csv`)，序列长度小于等于40，没有"抗菌"的标签。为了训练ranking和regression模型，我们给所有的负样本标上了一个模拟的MIC值，8192 μg/mL。采用了75:15:10的比例进行训练/验证/测试划分。为了数值调整，我们对输出目标空间进行了log10操作以进行平滑处理。

- Wet-lab dataset: 由于实验和环境的差异，Grampa数据集很可能存在噪声和偏差。因此，我们随机选择了67个来自正样本集的肽，并验证了它们的湿实验。事实上，我们发现其中一些肽的MIC值与Grampa数据集提供的相对应值有些不同。因此，我们使用这67个来自我们内部来源的正样本 (`data/origin_data/new_data_67.csv` / `data/filtered_data/experimental.csv`) 来进一步微调我们的模型，采用增量学习框架。



## Peptide encoding

- structured or tabularized data: 应用一些基本的描述符和伪氨基酸组成（PseAAC）描述符。最终的描述符涵盖了基本字符、氨基酸组成、自相关性、物理化学组成、准序列顺序和伪氨基酸组成，得到了长度为676的描述符。
- word-embedding: 将数据集中的每个肽序列都转换为一个矩阵𝑚×𝑑维的实数空间，𝑚是给定肽的长度，𝑑是嵌入的维度。



## Empirical selection

- 肽的净电荷是正电荷：因此，正的氨基酸残基(AAR)的数量大于负的AAR的数量。
- 肽具有特殊的两亲结构：一般来说，有两种构建两亲结构的模型。在第一个模型中，所有亲水性AARs和疏水性AARs都聚集在两侧。而在第二种模型中，疏水AARs与亲水性AARs在每2-3个残基上穿插。所有满足的肽都是由python脚本 (`sequence_generated/sequence_generated.py`) 生成的。



## Classification Model

- 对于XGBoost-tab模型，我们采用了标准化的实现，并对所有潜在的超参数设置进行了网格搜索。
- 我们发现XGBoost-tab模型的较小深度可以改善相应的性能，最终我们选择了以下超参数：modeldepth=4，number-of-estimators=600，learning-rate=0.1。

### train and test

- 训练数据？？

- 各模型的测试结果如下（见表 `SI/41551_2022_991_MOESM6_ESM.xlsx` sheet a）

|             | Accuracy | F1 Score | Precision | Recall |
| ----------- | -------- | -------- | --------- | ------ |
| XGBoost-tab | 0.9723   | 0.929    | 0.9422    | 0.9209 |
| LSTM-seq    | 0.9738   | 0.937    | 0.9243    | 0.95   |
| CNN-seq     | 0.9681   | 0.9247   | 0.8958    | 0.9556 |
| RF-tab      | 0.8247   | 0.2537   | 1         | 0.1412 |



## Ranking Model

- 直接在原始搜索空间上运行Ranking模型的计算成本是非常高的，因此在漏斗中的第一阶段使用了粗粒度的Classification模型来有效地减少搜索空间。

### train and test

- 训练数据？？
- 测试数据用了177条序列，测试结果见表 `SI/41551_2022_991_MOESM6_ESM.xlsx` sheet d



## Regression Model

- LSTM-seq模型被选定为最终的回归模型，它在全局和top-k-MSE得分方面表现出色。
- 在训练过程中，采用了L2损失作为回归任务的损失函数。还引入了top-k MSE来更好地评估具有高抗菌活性的肽段的回归能力。具体来说，从测试集中选择了最小MIC标签的k个样本，然后计算这些样本的MSE，以获得top-k MSE的测量值。

### train and test

- 各模型的测试结果见表 `SI/41551_2022_991_MOESM6_ESM.xlsx` sheet e，用于测试的序列数据见表 `SI/41551_2022_991_MOESM6_ESM.xlsx` sheet f



## Incremental learning

- 通过增量学习将模型参数调整到适应当前实验室条件，以便更准确地预测MIC值。
- 增量学习过程采用了简单的微调方法，通过在源域和目标域之间进行领域适应，以最大程度地适应目标域。最终，通过网格搜索找到了最佳模型和过程的参数配置。



## Model evaluation

```
===XGBoost Classification Model===
xgb_classfier_Accuracy : 0.9722
xgb_classfier_F1-score : 0.9412

===XGBoost Ranking Model===
xgb_rank_model top50: 0.44
xgb_rank_model top100: 0.68
xgb_rank_model top500: 0.8

===LSTM Regression Model===
Best MSE Error all 0.265339
Best MSE Error pos: 0.762097
Best R2 Error all: 0.808578
Best R2 Error pos: -0.647839
Best Top 20: 2.585441
Best Top 60: 1.292800
```

