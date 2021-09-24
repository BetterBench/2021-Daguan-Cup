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
    return test_dataset, test, id2label


def pre_ensemble(models_path, test_df, test_dataset,submit_name):
    config = Config()
    ensemble_list = []
    for i, models_path in enumerate(path_li):
        print("正在测试{}模型".format(models_path))
        test_prelist = []
        for fold in range(10):
            print("[{}/10]fold".format(fold+1))
            test_D = data_generator(test_dataset, config)
            # 每个模型的
            PATH = './{}/bert_{}.pth'.format(models_path,fold)
            model = torch.load(PATH)
            model.eval()
            with torch.no_grad():
                test_logit = None
                for input_ids, input_masks, segment_ids, labels in tqdm(test_D, disable=True):
                    y_pred = model(input_ids, input_masks, segment_ids)
                    y_pred = F.softmax(y_pred, dim=1)
                    y_pred = y_pred.detach().to("cpu").numpy()
                    if test_logit is None:
                        test_logit = y_pred
                    else:
                        test_logit = np.vstack((test_logit, y_pred))
                test_prelist.append(test_logit)
        test_pre = np.sum(np.array(test_prelist), axis=0) / (np.array(test_prelist).shape[0])
        ensemble_list.append(test_pre)
    test_preds = np.sum(np.array(ensemble_list), axis=0) / (np.array(ensemble_list).shape[0])
    test_preds = np.argmax(test_preds, axis=1)
    pred_labels = [id2label[i] for i in test_preds]
    SUBMISSION_DIR = "submit"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    Name = "{}-ensemble".format(submit_name)
    submit_file = SUBMISSION_DIR+"/{}.csv".format(Name)

    pd.DataFrame({"id": test_df['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)

    print()


if __name__ == "__main__":

    path_li = ["models-bertlstm-5986", "models_bertlast4", "models-bertfor-5904"]
    test_dataset, test, id2label = build_data()
    pre_ensemble(path_li, test, test_dataset, "-".join(path_li))

