import pandas as pd
from sklearn import metrics
import xgboost as xgb




predict_xgb_classifer_file = "/Users/zhouzhenyi/Documents/SCIProject/PeptideTools/SMEP/SMEP/sample/data_for_search.csv"
train_xgb_file = "/Users/zhouzhenyi/Documents/SCIProject/PeptideTools/SMEP/SMEP/sample/classification_train_sample.csv"
test_xgb_file = "/Users/zhouzhenyi/Documents/SCIProject/PeptideTools/SMEP/SMEP/sample/classification_test_sample.csv"


def XgbClassify():
    model = xgb.XGBClassifier(max_depth=4, n_estimators=600, learning_rate=0.1, use_label_encoder=False,
                              objective="binary:logistic")
    return model


def get_train_xgb_classifier_data(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path, encoding="utf8", index_col=0)
    test_data = pd.read_csv(test_data_path, encoding="utf8", index_col=0)
    x_train = train_data.iloc[:, 1:-2].values  # 676维度的特征表示
    x_test = test_data.iloc[:, 1:-2].values
    y_train = train_data.iloc[:, -1].values  # type值，0表示无活性，1表示有活性
    y_test = test_data.iloc[:, -1].values
    return x_train, x_test, y_train, y_test


def get_xgb_classifier_model(train_xgb_file, test_xgb_file):
    x_train, x_test, y_train, y_test = get_train_xgb_classifier_data(train_xgb_file,
                                                                     test_xgb_file)  # train和test数据要分开准备
    xgb_cls_model = XgbClassify()
    xgb_cls_model.fit(x_train, y_train, eval_metric='auc')
    y_pred = xgb_cls_model.predict(x_test)
    y_true = y_test
    print("xgb_classfier_Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
    print("xgb_classfier_F1-score : %.4g" % metrics.f1_score(y_true, y_pred))
    return xgb_cls_model


xgb_cls_model = get_xgb_classifier_model(train_xgb_file, test_xgb_file)
data_test = pd.read_csv(predict_xgb_classifer_file, chunksize=1000, encoding="utf-8", low_memory=False)
for chunk in data_test:
    y = xgb_cls_model.predict(chunk.iloc[:, 1:].values)
    mask = [bool(x) for x in y]

    print(mask)
    break
