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
        self.batch_size = 256
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
    text = text.replace('，','')
    text = text.replace('！', '')
    text = text.replace('？', '')
    text = text.replace('。', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text


def generate_label(pred, test_df, id2label):
    test_pre = np.argmax(pred, axis=1)
    pred_labels = [id2label[i] for i in test_pre]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "pesudo_label_{}".format(len(test_df))
    submit_file = SUBMISSION_DIR+"/{}.csv".format(Name)

    pd.DataFrame({"id": list(range(len(test_df))), "text": test_df['text'], "label": pred_labels}).to_csv(
        submit_file, index=False)


def predict_label(pred, test_df, id2label):
    test_pre = np.argmax(pred, axis=1)
    pred_labels = [id2label[i] for i in test_pre]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "all_train_berforclass_{}".format(len(test_df))
    submit_file = SUBMISSION_DIR+"/{}.csv".format(Name)

    pd.DataFrame({"id": list(range(len(test_df))), "text": test_df['text'], "label": pred_labels}).to_csv(submit_file, index=False)
def merge():
    train_f = '/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv'
    train_p = '/media/mgege007/winType/DaGuan/data/pesudo_label_100000.csv'

    train1 = pd.read_csv(train_f)
    train2 = pd.read_csv(train_p)
    pseudo_train_data = pd.concat([train1, train2]).reset_index().drop(columns=['index']).sample(frac=1)
    train_path = '/media/mgege007/winType/DaGuan/data/pesudo_label_data_{}.csv'.format(len(pseudo_train_data))
    pseudo_train_data.to_csv(train_path,index=False)
def pre_generate():
    config = Config()
    train_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv'
    train = pd.read_csv(train_clean)
    id2label = list(train['label'].unique())
    test_dataset = []
    path = "/media/mgege007/新加卷/Compition_data/datagrand_2021_unlabeled_data.json"
    num = 0
    test_text = []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            while True and num < 100000:
                line_data = f.readline()
                num += 1
                print(num)
                if line_data:
                    data = json.loads(line_data)
                    sentence_data = preprocess_text(data['title']+" "+data['content']).strip()
                    test_dict = {}
                    test_dict['text'] = sentence_data
                    test_text.append(sentence_data)
                    test_dict['label'] = [-1]*35
                    test_dataset.append(test_dict)
                else:
                    break
        except Exception as e:
            print(e)
            f.close()
    test_D = data_generator(test_dataset, config)
    PATH = './models/bertforclass.pth'
    model = torch.load(PATH)
    model.eval()
    n = 0
    with torch.no_grad():
        train_logit = None
        for input_ids, input_masks, segment_ids, labels in tqdm(test_D, disable=True):
            print(n)
            n+=1
            y_pred = model(input_ids, input_masks, segment_ids)
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = y_pred.detach().to("cpu").numpy()
            if train_logit is None:
                train_logit = y_pred
            else:
                train_logit = np.vstack((train_logit, y_pred))

    generate_label(train_logit, test_text, id2label)


def pre_predict():
    config = Config()
    train_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv'
    test_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_test.csv'
    train = pd.read_csv(train_clean)
    test = pd.read_csv(test_clean)
    test["text"].progress_apply(lambda x: preprocess_text(x))
    id2label = list(train['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    test_dataset = []
    for i in tqdm(range(len(test))):
        test_dict = {}
        test_dict['text'] = test.loc[i, 'text']
        test_dict['label'] = [-1]*35
        test_dataset.append(test_dict)
    test_D = data_generator(test_dataset, config)
    PATH = './models/bertforclass.pth'
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

    predict_label(train_logit, test, id2label)
if __name__ == "__main__":
    # pre_generate()
    # merge()
    # '/media/mgege007/winType/DaGuan/data/pesudo_label_data_114009.csv'
    pre_predict()

