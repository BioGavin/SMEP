from dataset import PeptideDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import LstmNet
import torch
import argparse
import pandas as pd
from utils import df2list, get_dict, get_reverse_dict, evaluate_customized, get_train_xgb_classifier_data, \
    get_train_xgb_rank_featured_data
from torch import nn
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
from torch.backends import cudnn
import random
import xgboost
from sklearn import metrics

"""
Lstm regerssor train
"""


class LstmTrain():
    def __init__(self, train_file, test_file):
        self.net = LstmNet(embedding_dim, hidden_num, num_layer, bidirectional, dropout, get_dict())
        self.net.to(device)
        # 读取训练数据，数据共有3列，分别是[sequence, MIC, type]
        train_data = pd.read_csv(train_file, encoding="utf-8", index_col=0).reset_index(drop=True)
        train_data = df2list(train_data, "sequence", "MIC", "type", ngram_num, log_num)  # 将训练数据格式化
        # 构建数据集，一个数据单元由4部分构成，第一是序列的one-hot表示，定长50，不足补零；第二是log10MIC；第三是type值，0or1；第四是序列长度
        train_dataset = PeptideDataset(train_data)
        # 数据加载，决定输入网络的数据形式，每次输入batch_size个数据
        # 一个batch_size的数据内容为：[tensor(序列),
        #                           tensor(log10MIC),
        #                           tensor(type),
        #                           tensor(序列长度)]
        # 一个batch_size的数据维度为：[torch.Size([batch_size, 50]),
        #                           torch.Size([batch_size, 1]),
        #                           torch.Size([batch_size]),
        #                           torch.Size([batch_size])]
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

        test_data = pd.read_csv(test_file, encoding="utf-8", index_col=0).reset_index(drop=True)
        test_data = df2list(test_data, "sequence", "MIC", "type", ngram_num, log_num)
        test_dataset = PeptideDataset(test_data)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        self.opt = torch.optim.AdamW(self.net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def __call__(self, *args, **kwargs):

        loss_best_all = 10000
        loss_best_pos = 10000
        r2_best_all = -5
        r2_best_pos = -5
        top20mse_best = 99
        top60mse_best = 99
        test_loss_all = []
        test_loss_pos = []
        test_r2_all = []
        test_r2_pos = []
        top_20_mse = []
        top_60_mse = []
        for epoch in range(epoch_num):
            if epoch > 0 and epoch % de_epo == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] *= de_r
            loss_all = 0
            count = 0
            self.net.train()
            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                text = batch[0]  # torch.Size([batch_size, 50])
                label = batch[1]  # torch.Size([batch_size, 1])
                length = batch[3]  # torch.Size([batch_size])
                text = text.to(device)
                label = label.to(device)
                out = self.net(text, length)
                loss = self.loss_fn(out, label)
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()
                loss_all += loss
                count += 1
                if count % 100 == 0:
                    print("\r Epoch:%d,Loss:%f" % (epoch, loss_all / count))

            self.net.eval()
            predict = list([])
            label_o = list([])
            sequence = []
            for idx, batch in enumerate(self.test_dataloader):
                text = torch.LongTensor(batch[0])
                label = batch[1]
                length = batch[3]
                text = text.to(device)
                label = label.to(device)
                out = self.net(text, length)

                predict.extend(out.cpu().detach().numpy().reshape(1, -1)[0])
                label_o.extend(label.cpu().detach().numpy())
                for i in text.cpu().numpy():
                    temp = []
                    for j in i[i > 0]:
                        temp.append(get_reverse_dict()[j])
                    sequence.append("".join(temp))
            df = pd.DataFrame({"sequence": sequence, "label": label_o, "predict": predict})
            pred = []
            label = []

            for i in range(len(label_o)):
                if label_o[i] <= np.log10(8196) - 0.01:
                    label.append(label_o[i])
                    pred.append(predict[i])
            mse_result_all = mean_squared_error(label_o, predict)  # 所有数据的均方误差MSE
            mse_result_pos = mean_squared_error(label, pred)  # 阳性数据的均方误差MSE
            r2_result_all = r2_score(label_o, predict)  # 所有数据的决定系数r2
            r2_result_pos = r2_score(label, pred)  # 阳性数据的决定系数r2
            test_loss_all.append(mse_result_all)
            test_loss_pos.append(mse_result_pos)
            test_r2_all.append(r2_result_all)
            test_r2_pos.append(r2_result_pos)
            top20_mse, top60_mse, _, _ = evaluate_customized(self.net, label, pred)
            top_20_mse.append(top20_mse)
            top_60_mse.append(top60_mse)
            if mse_result_all <= loss_best_all:
                loss_best_all = mse_result_all
                """
                保存1
                """
                torch.save(self.net.state_dict(), "./params/regress_allmse_%s.pth" % (epoch))
                df.to_csv("./results/lstm_result/regress/result_mse_all.csv", index=False)
            if mse_result_pos <= loss_best_pos:
                loss_best_pos = mse_result_pos
                """
                保存2
                """
                df.to_csv("./results/lstm_result/regress/result_mse_pos.csv", index=False)
                torch.save(self.net.state_dict(), "./params/regress_posmse_%s.pth" % (epoch))
            if r2_result_all >= r2_best_all:
                r2_best_all = r2_result_all
            if r2_result_pos >= r2_best_pos:
                r2_best_pos = r2_result_pos
            if top20_mse <= top20mse_best:
                top20mse_best = top20_mse
            if top60_mse <= top60mse_best:
                top60mse_best = top60_mse
                torch.save(self.net.state_dict(), "./params/regress_top60mse_%s.pth" % (epoch))

            print("Top 20 MSE: ", top20_mse)
            print("Top 60 MSE: ", top60_mse)
            print("MSE_loss_all = %f" % (mse_result_all))
            print("R2_score_all = %f" % (r2_result_all))
            print("MSE_loss_pos = %f" % (mse_result_pos))
            print("R2_score_pos = %f" % (r2_result_pos))
            print(
                "\r Epoch: %d Best MSE Error all %f ; Best MSE Error pos: %f ; Best R2 Error all: %f ; Best R2 Error pos: %f ; Best Top 20: %f ;Best Top 60: %f" \
                % (epoch, loss_best_all, loss_best_pos, r2_best_all, r2_best_pos, top20mse_best, top60mse_best))


def train_xgboost_classfier(train_data_path, test_data_path):
    x_train0, x_test, y_train0, y_test = get_train_xgb_classifier_data(train_data_path, test_data_path)

    learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    max_depth = [4, 5, 6, 7, 8, 9, 10]
    n_estimators = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    best_score = 0
    for lr in learning_rate:
        for md in max_depth:
            for ne in n_estimators:
                model = xgboost.XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr, use_label_encoder=False,
                                              objective="binary:logistic")
                model.fit(x_train0, y_train0, eval_metric='auc')

                y_pred = model.predict(x_test)
                y_true = y_test
                score = metrics.f1_score(y_true, y_pred)

                if score > best_score:
                    best_score = score
                    print("best_score:", best_score)
                    best_parameters = {'learning_rate': lr, "max_depth": md, "n_estimators": ne}
                    print("best_parameters:", best_parameters)
    model = xgboost.XGBClassifier(max_depth=best_parameters["max_depth"], n_estimators=best_parameters["n_estimators"],
                                  learning_rate=best_parameters["learning_rate"], use_label_encoder=False,
                                  objective="binary:logistic")
    model.fit(x_train0, y_train0, eval_metric='auc')
    y_pred = model.predict(x_test)
    y_true = y_test
    print("xgb_classifier accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
    print("xgb_classifier f1-score : %.4g" % metrics.f1_score(y_true, y_pred))


def train_xgboost_rank(train_data_path, test_data_path):
    x_train0, x_test, y_train0, y_test, test_data = get_train_xgb_rank_featured_data(train_data_path, test_data_path)

    learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    max_depth = [4, 5, 6, 7, 8, 9, 10]
    n_estimators = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    best_score = 0
    for lr in learning_rate:
        for md in max_depth:
            for ne in n_estimators:
                model = xgboost.XGBRegressor(max_depth=md, n_estimators=ne, learning_rate=lr, use_label_encoder=False,
                                             objective="rank:pairwise")
                model.fit(x_train0, y_train0)
                df = pd.DataFrame(test_data["sequence"].copy(), columns=["sequence"])
                y_pred = model.predict(x_test)
                df["MIC"] = y_pred
                df.sort_values("MIC", inplace=True)
                true_sequecne_top100 = test_data.iloc[0:100, ]["sequence"]
                pred_sequence_top100 = df.iloc[0:100, ]["sequence"]
                score = 0
                for i in true_sequecne_top100:
                    for j in pred_sequence_top100:
                        if i == j:
                            score += 1
                top100 = score / 100

                if top100 > best_score:
                    best_score = top100
                    print("best_score:", best_score)
                    best_parameters = {'learning_rate': lr, "max_depth": md, "n_estimators": ne}
                    print("best_parameters:", best_parameters)
    model = xgboost.XGBRegressor(max_depth=best_parameters["max_depth"], n_estimators=best_parameters["n_estimators"],
                                 learning_rate=best_parameters["learning_rate"], use_label_encoder=False,
                                 objective="rank:pairwise")

    model.fit(x_train0, y_train0)
    df = pd.DataFrame(test_data["sequence"].copy(), columns=["sequence"])
    y_pred = model.predict(x_test)
    df["MIC"] = y_pred
    df.sort_values("MIC", inplace=True)
    for k in [50, 100, 500]:
        true_sequecne_topk = test_data.iloc[0:k, ]["sequence"]
        pred_sequence_topk = df.iloc[0:k, ]["sequence"]
        score = 0
        for i in true_sequecne_topk:
            for j in pred_sequence_topk:
                if i == j:
                    score += 1
        topk = score / k
        print(f"xgb_rank top{k}:", topk)


def main():
    if mode == "lstm":
        train = LstmTrain(args.train_lstm_file, args.test_lstm_file)
        train()
    elif mode == "xgb_rank":
        train_xgboost_rank(args.train_xgb_file, args.test_xgb_file)
    elif mode == "xgb_classifier":
        train_xgboost_classfier(args.train_xgb_file, args.test_xgb_file)
    else:
        print("mode is error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dn", "--device_num", type=str, default="cuda:0", help="device_num")
    parser.add_argument("-ed", "--embedding_dim", type=int, default=50, help="embedding_dim")
    parser.add_argument("-hn", "--hidden_num", type=int, default=128, help="hidden_num")
    parser.add_argument("-nl", "--num_layer", type=int, default=2, help="num_layer")
    parser.add_argument("-dp", "--dropout", type=float, default=0.7, help="dropout")
    parser.add_argument("-bd", "--bidirectional", type=str, default="False", help="bidirectional")
    parser.add_argument("-n", "--ngram_num", type=int, default=1, help="ngram_num")
    parser.add_argument("-lg", "--log_num", type=int, default=10, help="log_num")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=2 * 1e-3, help="learning_rate")
    parser.add_argument("-e", "--epoch_num", type=int, default=100, help="epoch_num")
    parser.add_argument("-de", "--decline_epoch", type=int, default=16, help="decline_epoch")
    parser.add_argument("-dr", "--decline_rate", type=float, default=0.5, help="decline_rate")
    parser.add_argument("-sd", "--seed", type=int, default=500, help="seed")
    parser.add_argument("-md", "--mode", type=str, default="lstm", choices=["lstm", "xgb_rank", "xgb_classifier"],
                        help="three mode:lstm,xgb_rank,xgb_classifier")
    # xgb_rank and xgb_classifier模型调参的数据
    parser.add_argument("--train_xgb_file", type=str,
                        default="path_to_featured_training_set", help="train_xgb_file")
    parser.add_argument("--test_xgb_file", type=str,
                        default="path_to_featured_testing_set", help="test_xgb_file")
    # 训练和测试lstm模型的数据
    parser.add_argument("-trf", "--train_lstm_file", type=str, default="path_to_sequence_training_set",
                        help="train_lstm_file")
    parser.add_argument("-tef", "--test_lstm_file", type=str, default="path_to_sequence_training_set",
                        help="test_lstm_file")

    args = parser.parse_args()
    print(vars(args))

    device = torch.device(args.device_num if torch.cuda.is_available() else "cpu")
    embedding_dim = args.embedding_dim
    hidden_num = args.hidden_num
    num_layer = args.num_layer
    dropout = args.dropout
    ngram_num = args.ngram_num
    log_num = args.log_num
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epoch_num = args.epoch_num
    de_epo = args.decline_epoch
    de_r = args.decline_rate
    seed = args.seed
    mode = args.mode
    """
       fix random seed
    """
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if args.bidirectional == "True":
        bidirectional = True
    else:
        bidirectional = False
    main()
