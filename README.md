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
python3 generate_sample.py
```

- Train classification model

```bash
mkdir params
mkdir -p results/xgb_classifier_result
python3 train.py -md xgb_classifier --train_xgb_file sample/classify_sample.csv --test_xgb_file sample/classify_sample.csv > results/xgb_classifier_result/xgb_classifier_train.log
```

- Train ranking model

```bash
mkdir -p results/xgb_rank_result
python3 train.py -md xgb_rank --train_xgb_file sample/classify_sample.csv --test_xgb_file sample/classify_sample.csv > results/xgb_rank_result/xgb_rank_train.log
```

- Train regression model

```bash
mkdir -p results/lstm_result/regress
python3 train.py -md lstm --train_lstm_file sample/regression_train_sample.csv --test_lstm_file sample/regression_test_sample.csv > results/lstm_result/lstm_result_train.log
```

- Incremental learning

```bash
python3 lstm_fine_tune.py --lstm_param_path params/regress_allmse_28.pth --train_file_path data/origin_data/new_data_67.csv --save_parm_path params/finetune

python3 lstm_fine_tune.py --lstm_param_path params/regress_allmse_28.pth --train_file_path data/origin_data/new_data_67.csv --save_parm_path ft
```

- Predict

```bash
# Generate structual data for sequences
python3 ../featured_data_generated/cal_pep_des.py

# predict
mkdir prediction_results
python3 predict.py --lstm_param_path params/finetune --result_save_path prediction_results --train_xgb_file sample/classification_train_sample.csv --test_xgb_file sample/classification_test_sample.csv --predict_xgb_classifier_file sample/data_for_search.csv --save_xgb_classify_result True > prediction_results/predict.log
```

In `prediction_results` folder:

`classifier_feature_data.csv`: Positive result data predicted by XGBoost classification model.

`xgboost_classify.txt`: Positive sequences predicted by XGBoost classification model.

`top_500.csv`: The top 500 sequences predicted by the XGBoost sorting model.

`lstm_result.csv`: MIC results predicted by LSTM model.
