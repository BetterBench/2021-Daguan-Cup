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


class Config2:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = "BertLstm"
        self.Stratification = False
        # '/Bert_pytorch/bert_model_800/'
        self.model_path = '/media/mgege007/winType/DaGuan/Pytorch-pretrain/Bert_pytorch/bert_model_650/'
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


def preprocess_text2(document):
    # 删除逗号
    text = str(document)
    text = text.replace('，', '35001')
    text = text.replace('！', '35002')
    text = text.replace('？', '35003')
    text = text.replace('。', '35004')
    test = text.replace(',', '35001')
#    text = text.replace('17281', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text
def ensemble(pred, test_df, id2label,models):
    test_preds = np.sum(np.array(pred), axis=0) / (np.array(pred).shape[0])
    test_preds = np.argmax(test_preds, axis=1)
    pred_labels = [id2label[i] for i in test_preds]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "{}_ensemble".format(models)
    submit_file = SUBMISSION_DIR+"/{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(submit_file, index=False)


def ensemble_TTA(pred, pred_tta,test_df, id2label, models):
    test_pred= np.sum(np.array(pred), axis=0) / (np.array(pred).shape[0])
    tta_pred = np.sum(np.array(pred_tta), axis=0) / (np.array(pred_tta).shape[0])

    total = []
    for i in range(tta_pred.shape[0]):
        t = test_pred[i]+tta_pred[i]
        total.append(t)
    all_preds = np.array(total)
    test_preds = np.argmax(all_preds, axis=1)
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
    train = pd.read_csv(train_clean)
    test = pd.read_csv(test_clean)
    train["text"].progress_apply(lambda x: preprocess_text(x))
    test["text"].progress_apply(lambda x: preprocess_text(x))
    id2label = list(train['label'].unique())
    test_dataset = []
    for i in tqdm(range(len(test))):
        test_dict = {}
        test_dict['text'] = test.loc[i, 'text']
        test_dict['label'] = [-1]*35
        test_dataset.append(test_dict)

    test_2 = pd.read_csv(test_clean)
    test_2["text"].progress_apply(lambda x: preprocess_text2(x))
    test_dataset_2 = []
    for i in tqdm(range(len(test))):
        test_dict = {}
        test_dict['text'] = test.loc[i, 'text']
        test_dict['label'] = [-1]*35
        test_dataset_2.append(test_dict)
    return test_dataset, test_dataset_2, test, id2label
def build_tta():
    train_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv'
    test_clean = '/media/mgege007/winType/DaGuan/data/ttatest.csv'
    train = pd.read_csv(train_clean)
    test = pd.read_csv(test_clean)
    train["text"].progress_apply(lambda x: preprocess_text(x))
    test["text"].progress_apply(lambda x: preprocess_text(x))
    id2label = list(train['label'].unique())
    test_dataset = []
    for i in tqdm(range(len(test))):
        test_dict = {}
        test_dict['text'] = test.loc[i, 'text']
        test_dict['label'] = [-1]*35
        test_dataset.append(test_dict)

    test_2 = pd.read_csv(test_clean)
    test_2["text"].progress_apply(lambda x: preprocess_text2(x))
    test_dataset_2 = []
    for i in tqdm(range(len(test))):
        test_dict = {}
        test_dict['text'] = test.loc[i, 'text']
        test_dict['label'] = [-1]*35
        test_dataset_2.append(test_dict)
    return test_dataset, test_dataset_2, test, id2label


def pre_ensemble(model_li_1, model_li_2, test_dataset, test_dataset2):
    config = Config()
    config_2 = Config2()
    test_prelist = []
    test_D = data_generator(test_dataset, config)
    test_D_2 = data_generator(test_dataset, config_2)
    for i,path in enumerate(model_li_1):
        # 每个模型的
        print("正在测试{}".format(path))
        PATH = './ensemble_model/{}.pth'.format(path)
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
        test_prelist.append(train_logit)
    for i, path in enumerate(model_li_2):
        # 每个模型的
        print("正在测试{}".format(path))
        PATH = './ensemble_model/{}.pth'.format(path)
        model = torch.load(PATH)
        model.eval()
        n = 0
        with torch.no_grad():
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(test_D_2, disable=True):
                print(n)
                n += 1
                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = F.softmax(y_pred, dim=1)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))
        test_prelist.append(train_logit)
        test_prelist.append(train_logit)
    return test_prelist 
def submit(pred,pred2,test_df, id2label):
     
    test_preds_merge = np.sum(pred, axis=0) / (pred.shape[0])
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_preds_merge2 = np.sum(pred2, axis=0) / (pred2.shape[0])
    test_pre_tensor2 = torch.tensor(test_preds_merge2)

    Len=len(test_preds_merge)
    total=[]
    print(Len)
    print(len(test_preds_merge2))
    for i in range(Len):
        t=test_preds_merge[i]+test_preds_merge2[i]
        total.append(t)
    total=np.array(total)
    print(len(total))
    test_pre_tensor3=torch.tensor(total) 
    print(test_pre_tensor3[0])  
    test_pre = torch.max(test_pre_tensor3, 1)[1]
    pred_labels = [id2label[i] for i in test_pre]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "tta"
    submit_file = SUBMISSION_DIR+"/submit_{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)
# 不加TTA
if __name__ == "__main__":
    
    model_li_1 = ["bertfor","bertlstm"]
    model_li_2 = ["model_0", "model_1", "model_2", "model_3"]
    # 不加ＴＴＡ
#     test_dataset, test_dataset2, test, id2label = build_data()
#     test_prelist = pre_ensemble(model_li_1, model_li_2, test_dataset, test_dataset2)
#     ensemble(np.array(test_prelist), test, id2label, "3bert-4model-checkpoint")
 
    # 加入ＴＴＡ
    test_dataset, test_dataset2, test, id2label = build_data()
    test_prelist_1 = pre_ensemble(model_li_1, model_li_2, test_dataset, test_dataset2)
    

    tta_dataset, tta_dataset2, test_tta, id2label = build_tta()
    test_prelist_2 = pre_ensemble(model_li_1, model_li_2, tta_dataset, tta_dataset2)
    ensemble_TTA(np.array(test_prelist_1),np.array(test_prelist_2), test, id2label, "TTA-3bert-4model-checkpoint")

    print()
