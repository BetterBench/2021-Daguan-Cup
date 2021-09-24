from NEZHA.modeling_nezha import *
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn, optim
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from model import *
from utils import *
import time
from tqdm import tqdm
import re
import json
tqdm.pandas()
os.environ['PYTHONHASHSEED'] = '0'  # 消除hash算法的随机性
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)


class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = "BertLstm"
        self.Stratification = False
        # '/Bert_pytorch/bert_model_800/'
        self.model_path = '/media/mgege007/winType/DaGuan/Pytorch-pretrain/Bert_pytorch/bert_model_1000/'

        self.num_class = 35
        self.dropout = 0.2
        self.MAX_LEN = 100
        self.epoch = 6
        self.learn_rate = 2e-5
        self.normal_lr = 1e-4
        self.batch_size = 64
        self.k_fold = 10
        self.seed = 42

        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')

        self.focalloss = False
        self.pgd = False
        self.fgm = True


def preprocess_text(document):

    # 删除逗号
    text = str(document)
    text = text.replace('，', '')
    text = text.replace('！', '')
    text = text.replace('？', '')
    text = text.replace('。', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text


def ensemble(pred,tta_pred, test_df, id2label, models):
    total = []
    for i in range(pred.shape[0]):
        t = pred[i]+tta_pred[i]
        total.append(t)
    test_preds = np.argmax(np.array(total), axis=1)
    pred_labels = [id2label[i] for i in test_preds]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "{}_ensemble".format(models)
    submit_file = SUBMISSION_DIR+"/{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)


def build_data():
    train_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv'
    test_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_test.csv'
    tta_clean = '/media/mgege007/winType/DaGuan/data/tta_test.csv'
    train = pd.read_csv(train_clean)
    test = pd.read_csv(test_clean)
    tta = pd.read_csv(tta_clean)
    train["text"].progress_apply(lambda x: preprocess_text(x))
    test["text"].progress_apply(lambda x: preprocess_text(x))
    tta["text"].progress_apply(lambda x: preprocess_text(x))
    id2label = list(train['label'].unique())
    test_dataset = []
    for i in tqdm(range(len(test))):
        test_dict = {}
        test_dict['text'] = test.loc[i, 'text']
        test_dict['label'] = [-1]*35
        test_dataset.append(test_dict)
    tta_dataset = []
    for i in tqdm(range(len(tta))):
        test_dict = {}
        test_dict['text'] = tta.loc[i, 'text']
        test_dict['label'] = [-1]*35
        tta_dataset.append(test_dict)
    return test_dataset, tta_dataset, test, id2label


def pre_ensemble(model_li, test_dataset):
    config = Config()
    test_D = data_generator(test_dataset, config)
    test_li =[]
    for i, path in enumerate(model_li):
        # 每个模型的
        print("正在测试{}".format(path))
        PATH = './models/{}.pth'.format(path)
        model = torch.load(PATH)
        model.eval()
        n = 0
        with torch.no_grad():
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(test_D, disable=True):
                print(n)
                n += 1
                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = F.softmax(y_pred, dim=1)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))
        test_li.append(train_logit)
    test_preds = np.sum(np.array(test_li), axis=0) / (np.array(test_li).shape[0])

    return test_preds


if __name__ == "__main__":
    # checkpoint 融合
    model_li = ["bertforclass", "nezha_all", "nezhalarge_all", "bertlastcls_all",
                "bertlastfourcls_all", "bertlasttwoclspooler_all", "bertlstm"]

    test_dataset, tta_dataset, test, id2label = build_data()
    test_arr = pre_ensemble(model_li, test_dataset)
    tta_arr = pre_ensemble(model_li, tta_dataset)
    # TTA 测试集数据增强
    ensemble(test_arr, tta_arr, test, id2label, "-".join(model_li)+"TTA")

    print()
