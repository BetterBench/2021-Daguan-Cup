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
tqdm.pandas()
seed = 1949
os.environ['PYTHONHASHSEED'] = '0'  # 消除hash算法的随机性
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = "BertLastFourCls"
        self.Stratification = False
        # '/Bert_pytorch/bert_model_800/'
        self.model_path = '/Bert_pytorch/bert_model_1000/'

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
    text = text.replace('17281', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text
# 10fold的模型预测，生成提交文件
def submit(pred, test_df, id2label):
    test_preds_merge = np.sum(pred, axis=0) / (pred.shape[0])
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor, 1)[1]
    pred_labels = [id2label[i] for i in test_pre]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "Bert_single_alltrain"
    submit_file = SUBMISSION_DIR+"/submit_{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)

# 单个模型的预测，生成提交文件
def submit_single(pred, test_df, id2label):
    test_pre = np.argmax(pred,axis=1)
    pred_labels = [id2label[i] for i in test_pre]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "Bert_single_alltrain"
    submit_file = SUBMISSION_DIR+"/{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)
# 10fold的模型预测
def fold10_predict():
    config = Config()
    # train_clean = '/data/datagrand_2021_train.csv'
    train_clean = 'data/datagrand_2021_train.csv'
    test_clean = 'data/datagrand_2021_test.csv'
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
    model_pre = []
    for fold in tqdm(range(config.k_fold)):
        PATH = './models/bert_{}.pth'.format(fold)
        #model = MODEL_CLASSES[config.model](config).to(config.device)
        model = torch.load(PATH)
        model.eval()
        with torch.no_grad():
            y_p = []
            y_l = []
            val_y = []
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(test_D, disable=True):
                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = F.softmax(y_pred,dim=1)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))
            model_pre.append(train_logit)

    submit(np.array(model_pre), test, id2label)

# 单个模型的预测
def single_predict():
    config = Config()
    # train_clean = '/data/datagrand_2021_train.csv'
    train_clean = 'data/datagrand_2021_train.csv'
    test_clean = 'data/datagrand_2021_test.csv'
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
    model_pre = []
    
    PATH = './models/bertforclass.pth'
    model = torch.load(PATH)
    model.eval()
    with torch.no_grad():
        train_logit = None
        for input_ids, input_masks, segment_ids, labels in tqdm(test_D, disable=True):
            y_pred = model(input_ids, input_masks, segment_ids)
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = y_pred.detach().to("cpu").numpy()
            if train_logit is None:
                train_logit = y_pred
            else:
                train_logit = np.vstack((train_logit, y_pred))
            
    submit_single(train_logit, test, id2label)
if __name__=="__main__":
    # 不划分验证集模型预测
    single_predict()
    # 划分验证集，10fold的模型预测
    fold10_predict()
