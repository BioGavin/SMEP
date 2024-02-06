import os.path
import time

import numpy as np

from featured_data_generated import cal_pep_des
import math
import pandas as pd


def filter_positive_data(data, target_bacterium):
    bool_filter = data["bacterium"].str.contains(target_bacterium)
    filter_data = data[bool_filter]

    filter_data = filter_data[filter_data["has_cterminal_amidation"]]
    # filter_data = filter_data[~filter_data["has_unusual_modification"]]

    # print(f"filter: {filter_data.shape}")
    return filter_data


def generate_positive_data(data):
    data_all = [[], [], []]
    data = data[~data["sequence"].str.contains("B|J|X|Z|O|U")]  # 删除含有B|X|Z|O|U的序列
    for i in data["sequence"].unique():

        if len(i) < 6 or len(i) > 50:
            continue

        log_num = 0
        count = 0
        for v in data[data["sequence"] == i]["value"]:
            log_num += math.pow(10, v)
            count += 1
        # data_all[1] 存储MIC的列表
        # print(log_num)
        # if log_num > 256:
        #     continue
        data_all[0].append(i)  # data_all[0] 存储序列的列表
        data_all[1].append(float(log_num / count))  # 相同序列的多个活性值取10^i求和，然后再除以相同序列的个数
        # data_all[2] 存储type的列表 type都为1表示有活性，用于处理grampa数据集
        data_all[2].append(1)
    # print(data_all)

    data_all = list(map(list, zip(*data_all)))
    # print(data_all)
    data = pd.DataFrame(data=data_all, columns=["sequence", "MIC", "type"])
    print(f"positive data shape:{data.shape}")
    return data


def generate_negative_data(negative_file_path):
    data_negative = pd.read_csv(negative_file_path, encoding="utf8")
    data_negative = data_negative[~data_negative["Sequence"].str.contains("B|J|X|Z|O|U")]  # 删除含有B|X|Z|O|U的序列
    data_negative.reset_index(drop=True, inplace=True)
    data = pd.DataFrame(columns=["sequence", "MIC", "type"])
    for i in range(data_negative.shape[0]):
        data = data.append({"sequence": data_negative["Sequence"][i], "MIC": 8192, "type": 0}, ignore_index=True)
    data = data[data["sequence"].apply(lambda x: len(x) > 5)]
    data.drop_duplicates(inplace=True)
    print(f"negative data shape: {data.shape}")
    return data


def concat_datasets(positive_file, negative_file):
    data_concat = pd.concat([positive_file, negative_file], ignore_index=True, axis=0)  # 默认纵向合并0 横向合并1
    data_concat = data_concat.sample(frac=1, random_state=None)
    data_concat.reset_index(drop=True, inplace=True)
    print(f"all data shape: {data_concat.shape}")
    return data_concat


def split_sample(sample, generate_example_path, model_type):
    num = len(sample)
    train_sample = sample[:int(0.8 * num)]
    test_sample = sample[int(0.8 * num):]
    train_sample.to_csv(os.path.join(generate_example_path, f"{model_type}_train_sample.csv"), encoding="utf8")
    test_sample.to_csv(os.path.join(generate_example_path, f"{model_type}_test_sample.csv"), encoding="utf8")


if __name__ == '__main__':
    # 输入
    grampa_path = "data/origin_data/grampa.csv"
    negative_file_path = "data/origin_data/origin_negative.csv"
    generate_example_path = "sample"

    data = pd.read_csv(grampa_path, encoding="utf8")
    data = filter_positive_data(data, "aureus")  # 筛选对aureus有抗菌活性的数据
    positive_sample = generate_positive_data(data)  # 对相同序列的MIC值进行合并
    negative_sample = generate_negative_data(negative_file_path)

    common_sequences = np.intersect1d(positive_sample['sequence'], negative_sample['sequence'])
    positive_sample = positive_sample[~positive_sample['sequence'].isin(common_sequences)]
    negative_sample = negative_sample[~negative_sample['sequence'].isin(common_sequences)]

    print(f"common sequences: {len(common_sequences)}")
    print(f"positive data shape: {positive_sample.shape}")
    print(f"negative data shape: {negative_sample.shape}")

    all_sample = concat_datasets(positive_sample, negative_sample)

    negative_sample.to_csv(os.path.join(generate_example_path, "negative.csv"), index=False)
    positive_sample.to_csv(os.path.join(generate_example_path, "positive.csv"), index=False)
    # Generate classifier sample
    print("generating classify sample.....")

    num = len(all_sample)
    print(f"all sample len: {num}")
    start = time.time()
    sequence = all_sample["sequence"]  # type pd.Series
    peptides = sequence.values.copy().tolist()  # type list
    result = all_sample["MIC"]
    type = all_sample["type"]
    output_path = os.path.join(generate_example_path, "classify_dataset.csv")
    featured_df = cal_pep_des.cal_pep(peptides, sequence, result, type)
    split_sample(featured_df, generate_example_path, "xgb")
    end = time.time()
    print(f"generate classify feature data cost time: {end - start}s")
    print("generate regression all sample")


    # generate regression for all sample
    split_sample(all_sample, generate_example_path, "lstm")
    print("regression sample is ok")
