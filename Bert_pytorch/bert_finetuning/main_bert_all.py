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
from transformers import  Adafactor
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from model import *
from utils import *
import time
import logging
from tqdm import tqdm
from torch.cuda import amp # 要求pytorch>=1.6
from utils import *
tqdm.pandas()
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logging.basicConfig(level=logging.DEBUG, filename="train.log",filemode='a')

from NEZHA.modeling_nezha import *
MODEL_CLASSES = {
    'BertForClass': BertForClass,
    'BertLstm': BertLstm,
    'BertLastCls': BertLastCls,
    'BertLastTwoCls': BertLastTwoCls,
    'BertLastTwoClsPooler': BertLastTwoClsPooler,
    'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
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

class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = 'BertLastTwoClsPooler'  # "BertForClass"
        self.Stratification = False
        self.model_path = 'Bert_pytorch/bert_model_1000/'

        self.num_class = 35
        self.dropout = 0.1
        self.MAX_LEN = 100
        self.epoch = 100
        self.learn_rate = 4e-5
        self.normal_lr = 1e-4
        self.batch_size = 32
        self.k_fold = 10
        self.seed = 42
        self.device = torch.device('cuda')
        self.optimizer = "AdamW"
        self.focalloss = False
        self.pgd = False
        self.fgm = False
        self.scheduler = "cosine_schedule_with_warmup"
        self.fp16 = True


def preprocess_text(document):

    # 删除逗号
    text = str(document)
    text = text.replace('，', '')
    text = text.replace('！', '')
    text = text.replace('17281', '')
    # 用单个空格替换多个空格
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    return text


config = Config()
os.environ['PYTHONHASHSEED']='0'#消除hash算法的随机性
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)


train_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv'
# train_clean = '/media/mgege007/winType/DaGuan/data/pseudo_train_data_18330.csv'
train = pd.read_csv(train_clean)
# train = train[:1001]
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
scaler = amp.GradScaler()

train_D = data_generator(train_dataset, config, shuffle=True)
model = MODEL_CLASSES[config.model](config).to(config.device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)


if config.pgd:
    pgd = PGD(model)
    K = 3

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
        {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': normal_params, 'lr': config.normal_lr},
    ]
else:
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
adam_epsilon = 1e-6 
if config.optimizer == "AdamW":
    optimizer = AdamW(optimizer_parameters, lr=config.learn_rate)
elif config.optimizer == "lookahead":
    optimizer = AdamW(optimizer_parameters, lr=config.learn_rate, eps=adam_epsilon)
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
                scale_parameter = False,
                relative_step=False,
                warmup_init=False,
            )
warmup_steps = int(len(train) / config.batch_size / 2)
if config.scheduler == "constant_schedule":
        scheduler = get_constant_schedule(optimizer)

elif config.scheduler == "constant_schedule_with_warmup":
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
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
PATH = './{}/{}_all.pth'.format(config.model,config.model)
save_model_path = './{}/'.format(config.model)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
score = []
train_len = 0
loss_num = []
for e in range(config.epoch):
    print('\n------------epoch:{}------------'.format(e))
    model.train()
    tq = tqdm(train_D,ncols=70,disable=True)
    last=time.time()
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
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
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
    print(f"微调第{e}轮耗时：{time.time()-last}")
    torch.save(model.module if hasattr(model, "module") else model, PATH)
    print("train_loss={} train_f1={}".format(np.mean(loss_num), np.mean(score)))
optimizer.zero_grad()

    



