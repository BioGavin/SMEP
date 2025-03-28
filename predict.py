from dataset import PredictDataset
from torch.utils.data import DataLoader
from models import LstmNet, XgbClassify, XgbRank
import torch
import argparse
import pandas as pd
from utils import get_dict, get_reverse_dict, get_train_xgb_classifier_data, get_train_xgb_rank_featured_data
from sklearn import metrics
import os


class Predict():
    def __init__(self, lstm_param_path, lstm_result_save_path, train_xgb_file, test_xgb_file,
                 predict_xgb_classifer_file):
        self.lstm_result_save_path = lstm_result_save_path
        if not os.path.exists(self.lstm_result_save_path):
            os.mkdir(self.lstm_result_save_path)
        self.predict_xgb_classifer_file = predict_xgb_classifer_file  # 用于预测的结构化的数据
        self.param_path = lstm_param_path
        self.train_xgb_file = train_xgb_file
        self.test_xgb_file = test_xgb_file

    def __call__(self, *args, **kwargs):
        # 1. XGBoost 分类模型预测
        pos_feature_data = self.xgb_classifier_predict()  # 利用 xgb 分类器预测获取阳性结果
        xgboost_classify = pos_feature_data["sequence"].values.tolist()
        with open(os.path.join(self.lstm_result_save_path, "xgboost_classify.txt"), "w") as f:
            for i in range(len(xgboost_classify)):
                f.write(xgboost_classify[i])
                f.write("\n")
            f.close()
        # 2. XGBoost 排序模型预测
        xgb_rank_model = self.get_xgb_rank_model()  # 获取 xgb 排序模型
        xgb_rank_result = xgb_rank_model.predict(pos_feature_data.iloc[:, 1:].values)  # 利用 xgb 排序模型预测 MIC 值
        dataframe = pd.DataFrame([])
        dataframe["sequence"] = pos_feature_data["sequence"].copy()
        dataframe["MIC"] = xgb_rank_result
        dataframe.sort_values("MIC", inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        self.sequence = dataframe["sequence"].values[0:500]
        dataframe.iloc[:500, :].to_csv(os.path.join(self.lstm_result_save_path, "top_500.csv"))
        # 3. LSTM 回归模型预测
        self.lstm_preidct()

    def xgb_classifier_predict(self):
        xgb_cls_model = self.get_xgb_classifier_model()  # 获取 xgb 分类器模型
        data_test = pd.read_csv(self.predict_xgb_classifer_file, chunksize=50000, encoding="utf-8", low_memory=False,
                                index_col=0)
        df = pd.DataFrame([])
        for chunk in data_test:
            y = xgb_cls_model.predict(chunk.iloc[:, 1:].values)
            mask = [bool(x) for x in y]  # 预测值y转为bool值，0为False，1为True
            data = chunk[mask]  # 筛选出阳性结果
            df = pd.concat([df, data])  # 合并所有阳性结果
            print(df.describe())  # 统计每列的平均值、标准差等数据
        df.reset_index(drop=True, inplace=True)  # 重置索引
        if save_xgb_classify_result:
            df.to_csv(os.path.join(self.lstm_result_save_path, "classifier_feature_data.csv"), index=False)
        return df  # 经分类模型预测的阳性结果，第一列为序列，后676列为特征值

    def get_xgb_classifier_model(self):
        x_train, x_test, y_train, y_test = get_train_xgb_classifier_data(self.train_xgb_file,
                                                                         self.test_xgb_file)  # train和test数据要分开准备
        xgb_cls_model = XgbClassify()
        xgb_cls_model.fit(x_train, y_train, eval_metric='auc')
        y_pred = xgb_cls_model.predict(x_test)
        y_true = y_test
        print("xgb_classfier_Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
        print("xgb_classfier_F1-score : %.4g" % metrics.f1_score(y_true, y_pred))
        return xgb_cls_model

    def get_xgb_rank_model(self):
        x_train, x_test, y_train, y_test, test_data = get_train_xgb_rank_featured_data(
            self.train_xgb_file,
            self.test_xgb_file)
        xgb_rank_model = XgbRank()
        xgb_rank_model.fit(x_train, y_train)
        df = pd.DataFrame(test_data["sequence"].copy(), columns=["sequence"])
        y_pred = xgb_rank_model.predict(x_test)
        df["MIC"] = y_pred
        df.sort_values("MIC", inplace=True)  # 将预测的MIC值从小到大排序
        df.reset_index(drop=True, inplace=True)
        for k in [50, 100, 500]:
            true_sequecne_topk = test_data.iloc[0:k, ]["sequence"]  # 真实的 Top k 序列
            pred_sequence_topk = df.iloc[0:k, ]["sequence"]  # 预测的 Top k 序列
            score = 0
            for i in true_sequecne_topk:
                for j in pred_sequence_topk:
                    if i == j:
                        score += 1
            topk = score / k
            print(f"xgb_rank_model top{k}:", topk)  # topk 表示预测的前k条序列中有多少是在真实的前k条序列中的
        return xgb_rank_model

    def lstm_preidct(self):
        net = LstmNet(embedding_dim, hidden_num, num_layer, bidirectional, dropout, get_dict())
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(self.param_path, map_location=lambda storage, loc: storage.cuda(device_num)))
        else:
            net.load_state_dict(torch.load(self.param_path, map_location=torch.device('cpu')))
        net.to(device)
        net.eval()
        # 构建数据集，dataset中每个单元由2个元素的元组构成，第一个是对序列的one-hot编码，第二个是序列的长度。
        dataset = PredictDataset(self.sequence)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        sequence = []
        predict = []
        for i, (text, length) in enumerate(dataloader):
            text = text.to(device)
            out = net(text, length).reshape(-1)
            predict.extend(out.cpu().detach().numpy())
            for i in text:
                temp = ''.join([get_reverse_dict()[n] for n in i.tolist() if n > 0])
                sequence.append(temp)
        df = pd.DataFrame({"sequence": sequence, "predict": predict})
        df.sort_values("predict", inplace=True)
        # print(df)
        df.to_csv(os.path.join(self.lstm_result_save_path, "lstm_result.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--device_num", type=str, default=0)
    parser.add_argument("-ed", "--embedding_dim", type=int, default=50)
    parser.add_argument("-hn", "--hidden_num", type=int, default=128)
    parser.add_argument("-nl", "--num_layer", type=int, default=2)
    parser.add_argument("-dp", "--dropout", type=float, default=0.7)
    parser.add_argument("-bd", "--bidirectional", type=str, default="False")
    parser.add_argument("-scr", "--save_xgb_classify_result", type=str, default="False")
    parser.add_argument("-n", "--ngram_num", type=int, default=1)
    parser.add_argument("-lg", "--log_num", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-lr", "--learning_rate", type=float, default=2 * 1e-3)
    parser.add_argument("-en", "--epoch_num", type=int, default=100)
    parser.add_argument("-f", "--fmin", type=int, default=1)
    parser.add_argument("-lpp", "--lstm_param_path", type=str, default="path_to_params")
    parser.add_argument("-sfp", "--result_save_path", type=str, default="path_to_save_prediction_results")
    parser.add_argument("--train_xgb_file", type=str,
                        default="path_to_featured_training_set")
    parser.add_argument("--test_xgb_file", type=str,
                        default="path_to_featured_test_set")
    parser.add_argument("--predict_xgb_classifier_file", type=str,
                        default="path_to_the_featured_sequence_file_for_searching")
    args = parser.parse_args()
    print(vars(args))
    device_num = args.device_num
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    embedding_dim = args.embedding_dim
    hidden_num = args.hidden_num
    num_layer = args.num_layer
    dropout = args.dropout
    ngram_num = args.ngram_num
    log_num = args.log_num
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epoch_num = args.epoch_num
    fmin = args.fmin
    if args.bidirectional == "True":
        bidirectional = True
    else:
        bidirectional = False
    if args.save_xgb_classify_result == "True":
        save_xgb_classify_result = True
    else:
        save_xgb_classify_result = False

    predict = Predict(args.lstm_param_path, args.result_save_path, args.train_xgb_file,
                      args.test_xgb_file, args.predict_xgb_classifier_file)
    predict()
