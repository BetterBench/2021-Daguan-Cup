import pandas as pd
import numpy as np
import re
import torch
from transformers.optimization import get_constant_schedule
from sklearn.model_selection import KFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings
import torch.nn as nn
from tqdm import tqdm
import random
import gensim
import argparse
import os
from torch.utils import data
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from utils.adversarial_model import FGM, PGD
from utils.init_net import init_network
from utils.optimizer_lookahead import Lookahead
from utils.DL_model import *
torch.set_printoptions(edgeitems=768)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODELS = {
    'CapsuleNet':  CapsuleNet,
    'HAN':  HAN,
    'DPCNN':  DPCNN,
    'TextRCNNAttn':  TextRCNNAttn,
    'TextRCNN': TextRCNN
}
def basic_setting(SEED, DEVICE):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE != 'cpu':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#  数据预处理和训练词向量
def data_process():
    train_data = pd.read_csv("data/datagrand_2021_train.csv")
    test_data = pd.read_csv("/data/datagrand_2021_test.csv")
    id2label = list(train_data['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    y_train = np.zeros((len(train_data), len(id2label)), dtype=np.int8)

    all_sentences = pd.concat(
        [train_data['text'], test_data['text']]).reset_index(drop=True)
    all_sentences.drop_duplicates().reset_index(drop=True, inplace=True)
    all_sentences = all_sentences.apply(lambda x: x.split(' ')).tolist()
    if not os.path.exists('./embedding/w2v.model'):
        w2v_model = gensim.models.word2vec.Word2Vec(
            all_sentences, sg=1, vector_size=300, window=7, min_count=1, negative=3, sample=0.001, hs=1, seed=452)
        w2v_model.save('./embedding/w2v.model')
    else:
        w2v_model = gensim.models.word2vec.Word2Vec.load(
            "./embedding/w2v.model")

    if not os.path.exists('./embedding/fasttext.model'):
        fasttext_model = gensim.models.FastText(
            all_sentences, seed=452, vector_size=100, min_count=1, epochs=20, window=2)
        fasttext_model.save('./embedding/fasttext.model')
    else:
        fasttext_model = gensim.models.word2vec.Word2Vec.load(
            "./embedding/fasttext.model")
    train_dataset = []
    ylabel = []
    for i in tqdm(range(len(train_data))):
        train_dict = {}
        train_dict['text'] = train_data.loc[i, 'text']
        y_train[i][label2id[train_data.loc[i, 'label']]] = 1
        train_dict['label'] = y_train[i]
        ylabel.append(train_data.loc[i, 'label'])
        train_dataset.append(train_dict)
    test_dataset = []
    for i in tqdm(range(len(test_data))):
        test_dict = {}
        test_dict['text'] = test_data.loc[i, 'text']
        test_dict['label'] = -1
        test_dataset.append(test_dict)
    return test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label, ylabel


class DataSet(data.Dataset):
    def __init__(self, args, data, mode='train'):
        self.data = data
        self.mode = mode
        self.dataset = self.get_data(self.data, self.mode)

    def get_data(self, data, mode):
        dataset = []
        global s
        for data_li in tqdm(data):
            text = data_li['text'].split(' ')
            text = [w2v_model.wv.key_to_index[s] +
                    1 if s in w2v_model.wv else 0 for s in text]
            if len(text) < args.MAX_LEN:
                text += [0] * (args.MAX_LEN - len(text))
            else:
                text = text[:args.MAX_LEN]
            label = data_li['label']
            dataset_dict = {'text': text, 'label': label}
            dataset.append(dataset_dict)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        text = torch.tensor(data['text'])
        if self.mode == 'test':
            return text
        else:
            label = torch.tensor(data['label'])
            return text, label

# 封装数据集
def get_dataloader(args, dataset, mode):
    torchdata = DataSet(args, dataset, mode=mode)
    if mode == 'train':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    elif mode == 'test':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size*2, shuffle=False, num_workers=0, drop_last=False)
    elif mode == 'valid':
        dataloader = torch.utils.data.DataLoader(
            torchdata, batch_size=args.batch_size*2, shuffle=False, num_workers=0, drop_last=True)
    return dataloader, torchdata


loss_fun = nn.BCEWithLogitsLoss()
def validation_funtion(model, valid_dataloader, valid_torchdata, mode='valid'):
    model.eval()
    pred_list = []
    labels_list = []
    if mode == 'valid':
        for i, (description, label) in enumerate(tqdm(valid_dataloader)):
            output = model(description.to(DEVICE))
            pred_list.extend(output.sigmoid().detach().cpu().numpy())
            labels_list.extend(label.detach().cpu().numpy())
        labels_arr = np.array(labels_list)
        pred_arr = np.array(pred_list)
        labels = np.argmax(labels_arr, axis=1)
        pred = np.argmax(pred_arr, axis=1)
        auc = f1_score(labels, pred, average='macro')
        loss = loss_fun(torch.FloatTensor(labels_arr),
                        torch.FloatTensor(pred_arr))
        return auc, loss
    else:
        for i, (description) in enumerate(tqdm(valid_dataloader)):
            output = model(description.to(DEVICE))
            pred_list += output.sigmoid().detach().cpu().numpy().tolist()
        return pred_list


def train(args, fold, model, train_dataloader, valid_dataloader, valid_torchdata, model_num,early_stop=None):

    param_optimizer = list(model.named_parameters())
    embed_pa = ['embedding.weight']
    # 不训练embedding层
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in embed_pa)]},
                                    {'params': model.embedding.parameters(), 'lr': 5e-5}]
    num_train_steps = int(len(train_dataloader) * args.epochs)
    if args.optimizer == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr, amsgrad=True, weight_decay=5e-4)
    elif args.optimizer == "lookahead":
        optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.adam_epsilon)
        # optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr, amsgrad=True, weight_decay=5e-4)
        optimizer = Lookahead(optimizer=optimizer, la_steps=5, la_alpha=0.6)
    if args.scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5, last_epoch=-1)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=1e-5, last_epoch=-1)
    train_loss = []
    best_val_loss = np.inf
    best_f1 = 0
    best_loss = np.inf
    no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        if args.Model == "HAN" and epoch > 2:
            for param in model.named_parameters():
                if param[0] == 'embedding.weight':
                    param[1].requires_grad = True
                    break
        bar = tqdm(train_dataloader)
        # 遍历训练集
        for i, (description, label) in enumerate(bar):
            optimizer.zero_grad()
            output = model(description.to(DEVICE), label.to(DEVICE))
            loss = output
            loss.backward()
            train_loss.append(loss.item())
            scheduler.step()
            optimizer.step()
        # 遍历验证集
        f1, val_loss = validation_funtion(model, valid_dataloader, valid_torchdata, 'valid')
        print('Epoch:[{}/{}] train_loss: {:.5f}, val_loss: {:.5f},f1-score: {:.5f}\n'.format(
            epoch+1, args.epochs, np.mean(train_loss), val_loss, f1))

        if early_stop:
            if f1 > best_f1:
                best_val_loss = val_loss
                best_f1 = f1
                best_loss = train_loss[-1]
                torch.save(model.state_dict(), './saved/{}_model_{}.bin'.format(args.Model, fold))
            else:
                no_improve += 1
            if no_improve == early_stop:
                break
        else:
            if epoch >= args.epochs-1:
                # 保存模型权重
                torch.save(model.state_dict(), './saved/{}_model_{}.bin'.format(args.Model, fold))
    print('Fold:[{}/{}] best_trainloss: {:.5f}, best_valloss: {:.5f},best_f1score: {:.5f}\n'.format(
        fold, args.FOLD, best_loss, best_val_loss, best_f1))
    return best_val_loss, best_f1, best_loss


def run(args, train_dataset, w2v_model, fasttext_model, ylabel):
    kf = StratifiedKFold(n_splits=args.FOLD, shuffle=True, random_state=args.SEED)
    # kf = MultilabelStratifiedKFold(n_splits=args.FOLD, shuffle=True, random_state=2021)
    val_loss = []
    best_f1 = []
    train_loss = []
    model_num = 1
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(len(train_dataset)), ylabel)):
        model = MODELS[args.Model](args, w2v_model.wv.vectors.shape[0]+1, w2v_model.wv.vectors.shape[1]+fasttext_model.wv.vectors.shape[1], embeddings=True)
        #init_network(model)
        model.to(DEVICE)
        print(str(i+1), '-'*50)
        tra = [train_dataset[index] for index in train_index]
        val = [train_dataset[index] for index in test_index]
        print(len(tra))
        print(len(val))
        train_dataloader, train_torchdata = get_dataloader(args, tra, mode='train')
        valid_dataloader, valid_torchdata = get_dataloader(args, val, mode='valid')
        valloss, f1, trainloss = train(args, i, model, train_dataloader,
                                             valid_dataloader,
                                             valid_torchdata,
                                             model_num,
                                             early_stop=args.early_stop)
        torch.cuda.empty_cache()
        val_loss.append(valloss)
        train_loss.append(trainloss)
        best_f1.append(f1)
    # 打印每fold中最佳的模型的f1和loss
    for i in range(args.FOLD):
        print('- 第{}折中，best valloss: {}   best f1: {}   best trainloss: {}'.format(i +1, val_loss[i], best_f1[i], train_loss[i]))
    

# 生成提交文件
def get_submit(args, test_data, test_dataset, id2label):
    model = MODELS[args.Model](
        args, w2v_model.wv.vectors.shape[0]+1, w2v_model.wv.vectors.shape[1]+fasttext_model.wv.vectors.shape[1], embeddings=True)
    model.to(DEVICE)
    test_preds_total = []
    test_dataloader, test_torchdata = get_dataloader(args, test_dataset, mode='test')
    for i in range(0, args.FOLD):
        model.load_state_dict(torch.load('./saved/{}_model_{}.bin'.format(args.Model, i)))
        test_pred_results = validation_funtion(
            model, test_dataloader, test_torchdata, 'test')
        test_preds_total.append(test_pred_results)
    test_preds_merge = np.sum(test_preds_total, axis=0) / (args.FOLD)
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor, 1)[1]

    pred_labels = [id2label[i] for i in test_pre]
    submit_file = "./submit/{}.csv".format(args.Model)
    pd.DataFrame({"id": test_data['id'], "label": pred_labels}).to_csv(submit_file, index=False)


def arg_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', default='TextRCNN', type=str, help="")
    # Model可选择：CapsuleNet、HAN、DPCNN、TextRCNNAttn、TextRCNN
    parser.add_argument('--MAX_LEN', default=100, type=int,help='max length of sentence')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--SEED', default=9797, type=int, help='')
    parser.add_argument('--FOLD', default=10, type=int, help="k fold")
    parser.add_argument('--epochs', default=40, type=int, help="")
    parser.add_argument('--early_stop', default=40, type=int, help="")
    parser.add_argument('--lr', default=1e-3, type=float, help="")
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts', type=str, help="")
    parser.add_argument('--optimizer', default='AdamW', type=str, help="")
    parser.add_argument('--adam_epsilon', default=1e-6, type=float, help="")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 设置基本参数
    args = arg_setting() 
    # 设置随机种子
    basic_setting(args.SEED, DEVICE)  
    # 数据预处理
    test_data, train_dataset, test_dataset, w2v_model, fasttext_model, id2label, ylabel = data_process()
    # 开始训练
    run(args, train_dataset, w2v_model, fasttext_model, ylabel)
    # 获得提交结果文件
    get_submit(args, test_data, test_dataset, id2label)
