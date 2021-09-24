from NEZHA.modeling_nezha import *
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
import re
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from transformers import BertTokenizer, Adafactor, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from model import *
from utils import *
from utils import Lookahead
import time
import logging
from tqdm import tqdm
from torch.cuda import amp  # 要求pytorch>=1.6
tqdm.pandas()


class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = 'NEZHA'  # "BertForClass"
        self.Stratification = False
        self.model_path = 'Nezha_pytorch/nezha_model/'

        self.num_class = 35
        self.dropout = 0.2
        self.MAX_LEN = 100
        self.epoch = 50
        self.learn_rate = 4e-5
        self.normal_lr = 1e-4
        self.batch_size = 32
        self.k_fold = 10
        self.seed = 42
        self.device = torch.device('cuda')
        self.optimizer = "AdamW"
        self.focalloss = False
        self.pgd = False
        self.fgm = True
        self.scheduler = "cosine_schedule_with_warmup"
        self.fp16 = True


MODEL_CLASSES = {
    'BertForClass': BertForClass,
    'BertLstm': BertLstm,
    'BertLastCls': BertLastCls,
    'BertLastTwoCls': BertLastTwoCls,
    'BertLastTwoClsPooler': BertLastTwoClsPooler,
    'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
    'BertForClass_MultiDropout': BertForClass_MultiDropout,
    'BertLastTwoEmbeddingsPooler': BertLastTwoEmbeddingsPooler,
    'BertLastFourCls': BertLastFourCls,
    'BertLastFourClsPooler': BertLastFourClsPooler,
    'BertLastFourEmbeddings': BertLastFourEmbeddings,
    'BertLastFourEmbeddingsPooler': BertLastFourEmbeddingsPooler,
    'BertDynCls': BertDynCls,
    'BertDynEmbeddings': BertDynEmbeddings,
    'BertRNN': BertRNN,
    'BertCNN': BertCNN,
    'BertRCNN': BertRCNN,
    'XLNet': XLNet,
    'Electra': Electra,
    'NEZHA': NEZHA,

}

def preprocess_text(document):
    # 将符号替换为不在脱敏文本的词典中的词
    # 删除逗号, 脱敏数据中最大值为30357
    text = str(document)
    text = text.replace('，', '35001')
    text = text.replace('！', '35002')
    text = text.replace('？', '35003')
    text = text.replace('。', '35004')
    # text = text.replace('17281', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text

config = Config()
os.environ['PYTHONHASHSEED'] = '0'  # 消除hash算法的随机性
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# 数据预处理和加载
train_clean = 'data/datagrand_2021_train.csv'
train = pd.read_csv(train_clean)
train["text"].progress_apply(lambda x: preprocess_text(x))
ylabel = []
id2label = list(train['label'].unique())
label2id = {id2label[i]: i for i in range(len(id2label))}
y_train = np.zeros((len(train), len(id2label)), dtype=np.int8)
train_dataset = []
for i in tqdm(range(len(train))):
    train_dict = {}
    train_dict['text'] = train.loc[i, 'text']
    y_train[i][label2id[train.loc[i, 'label']]] = 1
    train_dict['label'] = y_train[i]
    ylabel.append(train.loc[i, 'label'])
    train_dataset.append(train_dict)

# K折划分
kf = StratifiedKFold(n_splits=config.k_fold, shuffle=True,
                     random_state=config.seed)

# FP16 混合精度训练
scaler = amp.GradScaler()
for fold, (train_index, valid_index) in enumerate(kf.split(np.arange(len(train_dataset)), ylabel)):
    print('\n\n------------fold:{}------------\n'.format(fold))
    tra = [train_dataset[index] for index in train_index]
    val = [train_dataset[index] for index in valid_index]

    train_D = data_generator(tra, config, shuffle=True)
    val_D = data_generator(val, config)
    model = MODEL_CLASSES[config.model](config).to(config.device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    # 是否PGD对抗训练
    if config.pgd:
        pgd = PGD(model)
        K = 3
    # 是否FGM对抗训练
    elif config.fgm:
        fgm = FGM(model)

    if config.focalloss:
        loss_fn = FocalLoss(config.num_class)
    else:
        loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步

    num_train_steps = int(len(train) / config.batch_size * config.epoch)
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if config.Stratification:
        bert_params = [x for x in param_optimizer if 'bert' in x[0]]
        normal_params = [p for n, p in param_optimizer if 'bert' not in n]
        optimizer_parameters = [
            {'params': [p for n, p in bert_params if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': normal_params, 'lr': config.normal_lr},
        ]
    else:
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
    # 优化器的选择
    adam_epsilon = 1e-6
    if config.optimizer == "AdamW":
        optimizer = AdamW(optimizer_parameters, lr=config.learn_rate)
    elif config.optimizer == "lookahead":
        optimizer = AdamW(optimizer_parameters,
                          lr=config.learn_rate, eps=adam_epsilon)
        optimizer = Lookahead(optimizer=optimizer, la_steps=5, la_alpha=0.6)

    elif config.optimizer == "Adafactor":
        optimizer = Adafactor(
            optimizer_parameters,
            lr=config.learn_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    # scheduler学习率的选择
    warmup_steps = int(len(train) / config.batch_size / 2)
    if config.scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)

    elif config.scheduler == "constant_schedule_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps)
    elif config.scheduler == "linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(train) / config.batch_size / 2),
            num_training_steps=num_train_steps
        )
    elif config.scheduler == "cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps,
        )

    elif config.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif config.scheduler == "polynomial_decay_schedule_with_warmup":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_train_steps,
        )

    best_f1 = 0
    best_epoch=-1
    best_valloss = 0
    # 每一个fold保存一个模型
    PATH = './{}_models/model_{}.pth'.format(config.model, fold)
    save_model_path = './{}_models/'.format(config.model)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    score = []
    train_len = 0
    loss_num = []
    for e in range(config.epoch):
        print('\n------------epoch:{}------------'.format(e))
        model.train()
        tq = tqdm(train_D, ncols=70, disable=True)
        last = time.time()
        for input_ids, input_masks, segment_ids, labels in tq:
            label_t = torch.tensor(labels, dtype=torch.float).to(config.device)
            if config.fp16:
                with amp.autocast():
                    y_pred = model(input_ids, input_masks, segment_ids)
                    loss = loss_fn(y_pred, label_t)
            else:
                y_pred = model(input_ids, input_masks, segment_ids)
                loss = loss_fn(y_pred, label_t)
            loss = loss.mean()
            if config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if config.pgd:
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    # 在embedding上添加对抗扰动, first attack时备份param.data
                    pgd.attack(is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    y_pred = model(input_ids, input_masks, segment_ids)

                    loss_adv = loss_fn(y_pred, label_t)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数

            elif config.fgm:
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                y_pred = model(input_ids, input_masks, segment_ids)
                loss_adv = loss_fn(y_pred, label_t)
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数

            # 梯度下降，更新参数
            if config.fp16:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            y_pred = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            label = np.argmax(labels, axis=1)
            score.append(f1_score(label, y_pred, average='macro'))
            loss_num.append(loss.item())
            tq.set_postfix(fold=fold, epoch=e,
                           loss=loss_num[-1], acc=score[-1])
        # 计算训练时间
        print(f"微调第{e}轮耗时：{time.time()-last}")
        # 验证集测试
        model.eval()
        with torch.no_grad():
            y_p = []
            y_l = []
            val_y = []
            val_loss = []
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(val_D, disable=True):
                label_t = torch.tensor(
                    labels, dtype=torch.float).to(config.device)
                y_pred = model(input_ids, input_masks, segment_ids)
                loss = loss_fn(y_pred, label_t)
                val_loss.append(loss.item())
                y_pred = F.softmax(y_pred, dim=1)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))

                y_p += list(y_pred[:, 1])

                y_pred = np.argmax(y_pred, axis=1)
                y_l += list(y_pred)
                y_label = np.argmax(labels, axis=1)
                val_y += list(y_label)

            val_f1 = f1_score(val_y, y_l, average="macro")
            if val_f1 >= best_f1:
                best_f1 = val_f1
                best_epoch = e
                best_valloss = np.mean(val_loss)
                torch.save(model.module if hasattr(
                    model, "module") else model, PATH)
        # 每一个epoch输出f1和loss
        print("fold [{}/{}] train_loss={} val_loss={} train_f1={} val_f1={}".format(fold +
              1, config.k_fold, np.mean(loss_num), np.mean(val_loss), np.mean(score), val_f1))
    # 打印每个fold中最佳f1的记录
    print("best_epoch={} val_loss={} val_f1={}".format(best_epoch,best_valloss,best_f1))
    optimizer.zero_grad()
