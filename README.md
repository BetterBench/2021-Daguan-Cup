# 【2021 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别】初赛Rank12的总结与分析

# 1 项目
（1）环境
> python 3.6+  
>pytorch 1.7.1+  
>cuda 11.2  
>transformers 4.9.2+  
>tqdm 4.61.2

（2）代码结构
```
├── Bert_pytorch # Bert 方案
│   ├── bert-base-chinese # 初始权重，下载地址https://huggingface.co/bert-base-chinese#
│   ├── bert_finetuning # Bert微调
│   │   ├── Config.py # Bert配置文件
│   │   ├── ensemble_10fold.py # 10折checkpoint融合
│   │   ├── ensemble_single.py #每种模型不划分验证集只生成的一个模型，用这些模型进行checkpoint融合
│   │   ├── generate_pseudo_label.py # 利用做高分模型 给无标注数据做伪标签
│   │   ├── main_bert_10fold.py # 划分10折的Bert，这种会存储10个模型，每一个fold一个模型
│   │   ├── main_bert_all.py # 不划分验证集的Bert，这种只会存储一个模型
│   │   ├── model.py # 17种魔改Bert，和其他网络的具体实现部分
│   │   ├── models 
│   │   ├── NEZHA # 网络结构实现文件，来源于官网
│   │   │   ├── configuration_nezha.py
│   │   │   └── modeling_nezha.py
│   │   ├── predict.py # 用模型模型进行预测测试集
│   │   ├── predict_tta.py # 用模型进行预测测试集，并使用TTA 测试集增强
│   │   ├── stacking.py # Stacking集成方法
│   │   └── utils.py # 工具函数
│   ├── bert_model_1000 # 存储预训练模型，下载地址https://drive.google.com/file/d/1rpWe5ec_buORvu8-ezvvAk9jrUZkOsIr/view?usp=sharing
│   ├── Data_analysis.ipynb # 数据分析
│   ├── Generate_TTA.ipynb # 生成TTA测试集增强的文件
│   └── pretrain # Bert预训练
│       ├── bert_model 
│       │   ├── vocab_100w.txt # 100W未标注数据语料的词典，有18544个词
│       │   ├── vocab_3462.txt # 整个训练集和测试集的词典，不包括未标注数据
│       │   └── vocab.txt
│       ├── NLP_Utils.py
│       ├── train_bert.py # Bert预训练主函数
│       └── transformers1.zip # transformes较高的版本
├── data
│   ├── datagrand_2021_test.csv # 测试集
│   └── datagrand_2021_train.csv # 训练集
├── Nezha_pytorch #NEZHA预训练方案
│   ├── finetuning #  Nezha微调
│   │   ├── Config.py　
│   │   ├── model.py　#模型实现文件
│   │   ├── models
│   │   ├── NEZHA
│   │   │   ├── configuration_nezha.py
│   │   │   └── modeling_nezha.py
│   │   ├── NEZHA_main.py　#微调主函数
│   │   ├── predict.py　# 10折模型预测
│   │   ├── submit
│   │   │   └── submit_bert_5epoch-10fold-first.csv
│   │   └── utils.py
│   ├── nezha-cn-base #nezha-base初始权重，下载地址https://github.com/lonePatient/NeZha_Chinese_PyTorch
│   ├── nezha_model #存放预训练生成的模型
│   ├── NEZHA_models
│   ├── nezha_output #预训练的checkpoint
│   ├── pretrain #nezha预训练
│   │   ├── __init__.py
│   │   ├── NEZHA
│   │   │   ├── configuration_nezha.py
│   │   │   ├── modeling_nezha.py
│   │   ├── nezha_model
│   │   │   └── vocab.txt # 预训练时，所需要的训练集的词典
│   │   ├── NLP_Utils.py
│   │   ├── train_nezha.py #预训练NEZHA的主函数
│   │   └── transformers1.zip # 更高版本的transformers
│   └── submit
```



# 2 引言



​	2021年的暑假，与博远、禹飞、沛恒刚打完科大讯飞的比赛，又续上类似的赛题2021的”达观杯“，继续经过一个多月，连续的战斗，比赛终于落下帷幕。A榜我们最高成绩可以达到0.62+，原本可以排名到第7，但是提交次数限制，未能提交最高得分文件。导致A榜只达到第12名。以及对于这种赛制的不理解，导致B榜滑落到21名。对我们的打击巨大。第一次打这种赛制的比赛，被恶心到了。但是也是学习到了很多东西，吸取教训，下次还能再接再厉。

​	该赛题和[2021年天池举办的全球人工智能大赛](https://tianchi.aliyun.com/competition/entrance/531852/introduction?spm=5176.12281957.1004.6.38b03eafApg5Vq)的赛道一几乎一样，就是标签性质不一样，天池赛题是多标签多分类，该赛题是多分类单标签。和[赛道三](https://tianchi.aliyun.com/competition/entrance/531851/information)也是类似，以及天池举办的新手赛-[新闻文本分类](https://tianchi.aliyun.com/competition/entrance/531810/introduction)都是一样的性质，脱敏数据的文本分类赛题。这个比赛我们参考了赛道一和赛道三的许多资料和方案。

​	7月26号已经开赛，8月16号这天才决定参赛，比赛花了36天。比赛过程分为三个阶段，第阶段钻研传统DL模型、第二阶段使用NEZHA和Bert实现预训练模型、第三阶段微调和预训练改进，以及多种提分技巧的应用。第一阶段，完全不需要云服务器，我本地的显卡就足够使用，但是来到第二阶段的开始使用预训练模型，我们必须使用恒源云上更大显存，更快运行速度的3090云服务器。苦战一个月，每天100左右的开销，邀请了所有周围的同学朋友，帮忙注册并认证，才送了不少的使用券，比赛的最后有一个星期，几个程序在跑，GPU不够用，我们成绩达到0.62+，排名也来得历史最高第三，租下了一个包周的3090连续跑了三天。队友还租的是恒源云的V100S，32G显存的显卡，跑nezha_large，都占满了，跑了好几天，开销巨大，预训练模型成本太高了，GPU也实在是太贵了。

# 3 方案

![3](https://pic3.zhimg.com/80/v2-d7ac2c039d7764ce56ffbf43d2a0508e_720w.jpg)

## 2.1 传统DL方案

详细的方案代码解析见[【2021 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别】3 DPCNN、HAN、RCNN等传统深度学习方案]()

我们的Baseline采用的是胶囊网络Capsule Net，进行线上提交就有0.55+的成绩，首次提交就排名在30+。传统的DL训练时间较短，仅有2小时左右。无标签的数据，我们未利用在传统的DL模型上。通过word2vec和fastext词向量的拼接、优化器的选择、sheduler学习率的选择，句子的最大长度选择截断，10折分层划分，用早停的方式去控制过拟合等方式，突破0.56+。同理实现DPCNN、RCNN、HAN，投票产生部分伪标签加入模型进行重新训练，单个模型DPCNN效果最高，达到传统DL模型的最高0.5779。再次投票得到0.5828的最高得分。单个模型最佳的参数如下

|    模型     | 词向量、维度         | Max_len | BS   | sheduler学习率               | 优化器 | Fold | 训练时间 |
| :---------: | -------------------- | ------- | ---- | ---------------------------- | ------ | ---- | -------- |
| Capsule Net | word2vc+fastext、128 | 100     | 32   | CosineAnnealingWarmRestrarts | Adamw  | 10   | 2.0小时  |
|    RCNN     | word2vc+fastext、128 | 100     | 32   | CosineAnnealingWarmRestrarts | Adamw  | 10   | 2.5小时  |
|     HAN     | word2vc+fastext、128 | 100     | 32   | CosineAnnealingLR            | Adamw  | 10   | 2.5小时  |
|    DPCNN    | word2vc+fastext、128 | 100     | 32   | CosineAnnealingWarmRestrarts | Adamw  | 10   | 2.0小时  |

对比过的选择

+ Scheduler学习率
  + Constant_shedule
  + CosineAnnealingWarmRestarts 最佳
  + CosineAnnealing 较好
+ 优化器
  + Lookhead
  + AdamW 最佳
+ 对抗训练
  + FGM 效果不佳
  + PGD 效果不佳
+ K折划分方式
  + Kfold
  + MutilabelStratifiedKfold
  + StratifiedKfold 最佳



## 2.2 预训练方案

详细代码解析见[【2021 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别】3 Bert和Nezha方案]()

这种方案花费了我们四个星期的时间，训练迭代优化过程非常缓慢，但效果显著。预训练和微调训练参数对应训练时间如下。（Batch size 简称BS）

| 模型        | 预训练Epoch | 预训练BS | 微调Epoch | 微调BS | 对抗训练 | GPU设备 |   训练时间   | 占用显存 |
| ----------- | :---------: | :------: | :-------: | :----: | :------: | :-----: | :----------: | :------: |
| 魔改Bert    |    1000     |    32    |    50     |   32   |    无    |  3090   | 12+7=19小时  |    7G    |
| Nezha-base  |     480     |    32    |    50     |   32   |   FGM    |  3090   | 6+13=19小时  |    7G    |
| Nezha-large |     300     |    64    |    50     |   32   |    无    |  V100S  | 4+9 = 13小时 |   31G    |

+ 总共预训练只训练14009+6004条样本数据。未标注数据，我们有加入40万的语料去训练NEZHA，用3090训练了5天5夜，用来微调测试的效果并不佳，时间成本太高最终放弃该方案。词典也尝试过用10W 语料1.8w+的词典大小，去预训练，发现线上效果都不如只使用标注数据词典的效果。最终还是选择3456个词个数的词典和只使用标注的训练集。

+ Bert模型

  + 模型并不是使用传统的bert，使用多种魔改的Bert方案，最终在Bert后接上一个LSTM，效果最佳，次之是最后一层向量取平均后与最后一层cls拼接的魔改Bert最佳
  + 其他魔改，比如只是用最后一层cls，最后四层cls拼接等等17种魔改Bert，具体见实现代码[model.py]()

+ Bert预训练的技巧有

  + 首尾截断方式：首部阶段和尾部截断方式并没有时间进行对比，预训练的调参时间成本太高。

  + 动态MASK 策略：可以每次迭代都随机生成新的mask文本，增强模型泛化能力。

  + Mask概率 ：0.15，在NEZHA上尝试加大训练难度，改为过0.5，但是在Bert上并没有带来增益

  + N-gram 掩码策略：以Mask掩码概率选中token，为增加训练难度，选中部分以70%、20%、10%的概率进行1-gram、2-gram、3-gram片段的mask（选中token使用[MASK]、随机词、自身替换的概率和原版Bert一致）

  + 权重衰退:weight_decay=0.01

  + Warmup学习率

  + 数据预处理：将逗号、感叹号、问号中文符号删除，且删除最高词频的17281。

    

+ NEZHA预训练技巧和Bert类似，区别在于数据预处理不同、掩码概率不同，选取的是0.5，且尝试了冻结word_Eembedding之外的所有层，只训练该层，加快预训练时间，缩短了一半的时间。但是这种冻结参数的方式，在Bert和nazha_large上，预训练的loss下降非常缓慢，最终只在nezha上进行了实验。

  + 数据预处理：并未删除中文符号，还将其替换为大于词典最大数的脱敏数据

  + Mask概率：0.5
  + 冻结word_Embedding以外的所有层数。

+ NEZHA和Bert的微调几乎类似，唯一的区别就是在于数据预处理的方式不一样，具体实现，查看[【2021 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别】3 Bert和Nezha方案]()

  

# 3 提分技巧

+ 训练集数据增广
  + 尝试过EDA的数据增广，效果并不佳
  + 在比赛后期，用在TTA测试集数据增强的上一种方式，还未在训练集上尝试，就是Shuffle每个样本的句子级别。把每个样本的句子进行调动顺序，并不是EDA中词级别的shuffle
+ 伪标签
  + 利用多模型方案投票的原理，选出测试集的高质量伪标签，加入训练集，重新训练模型。在此任务中，只在传统DL方案中有效果，在预训练方案中无效，反而降低了模型效果，具体原因分析，可能是因为该任务的本身计算的准确率只有60%不到。做出来的伪标签质量并不高。
  + 可以利用主办方提供的未标注数据，生成伪标签进行训练，但是由于该任务的准确率实在太低，A榜第一都只有0.63+的准确率，生成的伪标签质量并不高，这种方案在该任务中行不通。
+ 投票融合
  + 利用不同模型能学习到的不同特征，多模型的结果进行投票，能提升4个千分点。但是仅限于模型之间线上得分差异较小。比如我们Nezha单模达到了0.62+的时候，Bert和其他方案还在0.59+徘徊，这样的投票融合，反而会拉低最高单模的分数，加权也不行，血的教训。
+ checkpoint融合
  + 每个fold都存储一个模型，等程序跑完将这些模型一起加载都预测一遍测试集，得到多个6004行35列的矩阵，每行取算术平均后再进行计算标签。同样要求模型之间线上得分差异小，差异大了，加权也无法带来增益，反而会拉低最高单模的效果。具体实现，可以查看代码[predict.py]()
+ TTA测试集数据增强
  + 对测试集的样本进行句子级别的shuffle，作为新的测试集，用模型预测，得到6004行35列的矩阵，与原始测试集用模型预测得到6004行35列的矩阵，相加后取算术平均作为最终测试集的预测结果。线上能带来3个千分点的增益。具体实现，可以查看代码[predict.py]()

# 4 加快训练

+ FP16：混合精度训练，每个epoch加快10S左右
+ 预训练只训练word_embedding，在其他赛题的任务中，有人提出过冻结word_embedding和position_embedding，但是我们在复现该方法时，查看NEZHA模型只有word_embedding层，并未找到position_embedding层。训练时间缩短一半。但是，该方法在Bert上导致预训练Loss下降缓慢，甚至不下降，最终只在Nezha-base上尝试使用。
+ 用更高配置的GPU：我们通过使用不同的显卡设备，发现设备配置越高，训练速度越快，即使占用的显存都一样。3090真香。



# 5 总结和反思

（1）总结

+ 在比赛中，做预训练模型，选用初始设置跑出来一个预训练模型后，再去固定了微调方案，反过来去对预训练方案进行改进和调参。不要着急去做微调，我们这次的比赛中，就犯了这个错误，预训练方案到比赛的最后一天都没有最终确定下来，最后一天还在跑预训练。导致比赛的最后阶段没有去做好微调方案，还有很多微调方案没来得及尝试和对比。
+ 我们团队虽然使用了语雀来维护一个文档，但是代码并没有管理，导致经常出现队友之前代码不一致，沟通和任务安排经常出现偏差。应该使用Git去管我们的代码
+ 队友之间配合还欠缺默契，经常传递信息不够明确，过程中出现了，队友之间跑着一样的程序，占用着两个GPU，或者说用GPU跑着一个没有实验意义的程序。团队中还出现，跑的程序不知道和哪个程序是对比实验，跑出来的结果没有实验对比性，无法判断跑的某个点是否带来增益，白白浪费GPU和时间。

（2）继续提升方向

+ 预训练

  + 参考roberta，将句子复制若干份，让模型见到更多的句子遮罩方法，提高模型见到token的数量，提升预训练模型的鲁棒性
  + 句子数据增广后再预训练
  + TF-IDF Predict Task:提取TFIDF权重，显示的告诉模型权重，让模型学习不同词权重在中的分布关系（来源[[2021天池全球人工智能大赛赛道一](https://gaiic.tianchi.aliyun.com/)冠军方案提出）
  + 掩码策略改进（思路来源：https://github.com/nilboy/gaic_track3_pair_sim）
    + WWM 完全掩码
    + 动态Mask
    + n-gram Mask
    + 混合Maks 
    + similar ngram mask
  + 加入主办方提供的未标注数据，足足有72G，如果时间允许，设备足够高，预训练充分后，这将会带来巨大的增益。
  + 通过Bert实现同义词替换（思路来源：天池-全球人工智能大赛赛道一-rank2-炼丹术士）

  ![tong](https://pic1.zhimg.com/80/v2-d373d0ac9cdb1741c70d6c5f9fa4da04_720w.jpg)

+ 问题优化（思路来源:[小布助手问题匹配-冠军方案](https://yiwise-algo.yuque.com/docs/share/5a1e3b76-4d04-4127-979a-496d7bc8c1b8?#%20%E3%80%8A%E7%9F%AD%E6%96%87%E6%9C%AC%E7%9B%B8%E4%BC%BC%E5%8C%B9%E9%85%8D%E3%80%8B)）

- ![qe](https://pic3.zhimg.com/80/v2-2462951e02f28351f5821209b69fd2ae_720w.jpg)

- 微调
  + EDA 数据增广在脱敏数据上表现不佳，但是AEDA这个方法还未尝试过，就是在句子中随机插入标点符号。（来源资料：https://mp.weixin.qq.com/s/R6uDbn3CqxFkOye73Rqpqg）

+ 模型融合

  + Stacking：我实现过，单个模型都上了0.58+，但是本地验证只有0.55+左右，理论上不应该的，应该是未能正确实现
  + Checkpoint融合：这种方案得到的结果最为稳重，我们在B榜没有经验，提交的文件只是单模的，我们未能提交融合后的方案。

+ 伪标签

  + 由于该任务本身准确率不高，就连A榜第一都只有63%的准确率，做出来的标签不佳，但是如果在其他准确率高的任务中，这将会是一个大杀器。
  + 做伪标签的数据除了是测试集，还可以是未标注的数据，未标注的数据有足够大，足够训练模型。

+ 新方案

  + ELETRA-Pytorch版本，并没有尝试

      - https://github.com/richarddwang/electra_pytorch
      - https://github.com/lucidrains/electra-pytorch

  - 知识蒸馏的预训练模型 
    - 训练加速
    - 华为 TinyBert   fine-tune阶段采用了数据增强的策略(mask之后预测 并使用余弦相似度来选择对应的N个候选词最后以概率p选择是否替换这个单词，从而产生更多的文本数据)
  - 百度**ERNIE** pytorch
    - https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch
  - ConVBert
    - https://github.com/codertimo/ConveRT-pytorch
