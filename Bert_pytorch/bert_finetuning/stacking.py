from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import *
from model import *
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from NEZHA.modeling_nezha import *
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
import random
import os
import re
from tqdm import tqdm
tqdm.pandas()
os.environ['PYTHONHASHSEED'] = '0'  # 消除hash算法的随机性
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)

MODEL_CLASSES = {
    'BertForClass': BertForClass,
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
        self.model = "BertForClass"
        self.Stratification = False
        self.model_path = '/media/mgege007/winType/DaGuan/Pytorch-pretrain/Nezha_pytorch/nezha_model/'

        self.num_class = 35
        self.dropout = 0.2
        self.MAX_LEN = 100
        self.epoch = 6
        self.learn_rate = 2e-5
        self.normal_lr = 1e-4
        self.batch_size = 512
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


def build_data():
    train_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_train.csv'
    test_clean = '/media/mgege007/winType/DaGuan/data/datagrand_2021_test.csv'
    train = pd.read_csv(train_clean)
    test = pd.read_csv(test_clean)
    train["text"].progress_apply(lambda x: preprocess_text(x))
    test["text"].progress_apply(lambda x: preprocess_text(x))
    ylabel = []
    id2label = list(train['label'].unique())
    label2id = {id2label[i]: i for i in range(len(id2label))}
    y_train = np.zeros((len(train), len(id2label)), dtype=np.int8)
    test_dataset = []
    for i in tqdm(range(len(test))):
        test_dict = {}
        test_dict['text'] = test.loc[i, 'text']
        test_dict['label'] = [-1]*35
        test_dataset.append(test_dict)
    train_dataset = []
    for i in tqdm(range(len(train))):
        train_dict = {}
        train_dict['text'] = train.loc[i, 'text']
        y_train[i][label2id[train.loc[i, 'label']]] = 1
        train_dict['label'] = y_train[i]
        ylabel.append(train.loc[i, 'label'])
        train_dataset.append(train_dict)
    return train_dataset, test_dataset, ylabel, test, id2label


def pre_stacking(models_path, train_dataset, test_dataset, ylabel):
    config = Config()

    kf = StratifiedKFold(n_splits=config.k_fold,
                         shuffle=True, random_state=config.seed)
    val_pre = []
    val_label = []
    test_prelist = []
    val_logit = None
    for fold, (train_index, valid_index) in enumerate(kf.split(np.arange(len(train_dataset)), ylabel)):
        print('\n\n------------fold:{}------------\n'.format(fold))
        val = [train_dataset[index] for index in valid_index]
        test_D = data_generator(test_dataset, config)
        val_D = data_generator(val, config)
        # 每个模型的
        PATH = './{}/model_{}.pth'.format(models_path, fold)
        model = torch.load(PATH)
        model.eval()
        with torch.no_grad():
            y_p = []
            y_l = []
            val_y = []
            test_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(val_D, disable=True):
                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = F.softmax(y_pred, dim=1)
                y_pred = y_pred.detach().to("cpu").numpy()
                val_label.extend(list(np.argmax(np.array(labels), axis=1)))
                if val_logit is None:
                    val_logit = y_pred
                else:
                    val_logit = np.vstack((val_logit, y_pred))
            # val_pre.append(val_logit)

            for input_ids, input_masks, segment_ids, labels in tqdm(test_D, disable=True):
                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = F.softmax(y_pred, dim=1)
                y_pred = y_pred.detach().to("cpu").numpy()
                if test_logit is None:
                    test_logit = y_pred
                else:
                    test_logit = np.vstack((test_logit, y_pred))
            test_prelist.append(test_logit)
    test_pre = np.sum(np.array(test_prelist), axis=0) / \
        (np.array(test_prelist).shape[0])
    val_path = "val_data"
    test_path = "test_data"
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        os.makedirs(test_path)
    val_pre_name = "./val_data/{}_val_pre.npy".format(models_path)
    val_label_name = "./val_data/{}_val_label.npy".format(models_path)
    test_pre_name = "./test_data/{}_test_pre.npy".format(models_path)
    np.save(val_pre_name, np.array(val_logit))
    np.save(val_label_name, np.array(val_label))
    np.save(test_pre_name, np.array(test_pre))


def stacking(path_li, test_df, id2label):
    val_x = None
    val_y = []
    test_x = None

    for i, p in enumerate(path_li):
        val_pre_name = "./val_data/{}_val_pre.npy".format(p)
        test_pre_name = "./test_data/{}_test_pre.npy".format(p)
        val_label_name = "./val_data/{}_val_label.npy".format(p)
        val_pre = np.load(val_pre_name, allow_pickle=True)
        val_y.append(np.load(val_label_name, allow_pickle=True))
        test_pre = np.load(test_pre_name, allow_pickle=True)
        if val_x is None:
            val_x = val_pre
            test_x = test_pre
        else:
            val_x = np.hstack((val_x, val_pre))
            test_x = np.hstack((test_x, test_pre))
    scaler = StandardScaler()

    train_proba = val_x
    test_proba = test_x
    label = val_y[0]

    scaler.fit(train_proba)
    train_proba = scaler.transform(train_proba)
    test_proba = scaler.transform(test_proba)
    lr = LogisticRegression(tol=0.0001, C=0.5, random_state=98, max_iter=10000)

    kf = StratifiedKFold(n_splits=5, random_state=244, shuffle=True)
    pred_list = []
    score = []
    for fold, (train_index, val_index) in enumerate(kf.split(train_proba, label)):
        X_train = train_proba[train_index]
        y_train = label[train_index]
        X_val = train_proba[val_index]
        y_val = label[val_index]
        lr.fit(X_train, y_train)
        y_pred = lr.predict_proba(X_val)
        y_pred = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        score.append(f1)
        print("{} fold f1 = {}".format(fold+1, f1))
        y_testi = lr.predict_proba(test_proba)
        pred_list.append(y_testi)
    test_preds = np.sum(np.array(pred_list), axis=0) / \
        (np.array(pred_list).shape[0])
    test_preds = np.argmax(test_preds, axis=1)
    pred_labels = [id2label[i] for i in test_preds]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "NEZHA-Bert-Stacking"
    submit_file = SUBMISSION_DIR+"/{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)
    # print(lr.coef_, lr.n_iter_)
    print("最终平均得分={}".format(np.mean(score)))
    print()


if __name__ == "__main__":

    path_li = ["bert_model", "nezhalarge_model"]
    train_dataset, test_dataset, ylabel, test, id2label = build_data()
    for i, p in enumerate(path_li):
        pre_stacking(p, train_dataset, test_dataset, ylabel)
    stacking(path_li, test, id2label)
