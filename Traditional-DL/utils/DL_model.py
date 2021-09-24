
import gensim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from utils.spatial_dropout import SpatialDropout

class DPCNN(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, embeddings=None):
        super(DPCNN, self).__init__()
        # self.dropout = 0.1  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 35  # 类别数
        self.learning_rate = 1e-3  # 学习率
        self.num_filters = 250  # 卷积核数量(channels数)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        hidden_size = 128
        # 字向量维度
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if embeddings:
            w2v_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/w2v.model").wv
            fasttext_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/fasttext.model"
            ).wv
            w2v_embed_matrix = w2v_model.vectors
            fasttext_embed_matrix = fasttext_model.vectors
            #             embed_matrix = w2v_embed_matrix
            embed_matrix = np.concatenate(
                [w2v_embed_matrix, fasttext_embed_matrix], axis=1
            )
            oov_embed = np.zeros((1, embed_matrix.shape[1]))
            embed_matrix = torch.from_numpy(
                np.vstack((oov_embed, embed_matrix)))
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False
        self.spatial_dropout = SpatialDropout(drop_prob=0.5)
        self.conv_region = nn.Conv2d(
            1, self.num_filters, (3, self.embedding_dim))
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x, label=None):
        x = self.embedding(x)
        x = self.spatial_dropout(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        out = self.fc(x)
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px
        return x


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = nn.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(nn.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = nn.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = nn.tanh(eij)
        a = nn.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / nn.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * nn.unsqueeze(a, -1)  # [16, 100, 256]
        return nn.sum(weighted_input, 1)

#定义模型


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, True)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)
        x = a.mul(x).sum(1)
        return x


class HAN(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, embeddings=None):
        super(HAN, self).__init__()
        self.num_classes = 35
        hidden_size_gru = 256
        hidden_size_att = 512
        hidden_size = 128
        self.num_words = args.MAX_LEN
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embeddings:
            w2v_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/w2v.model").wv
            fasttext_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/fasttext.model").wv
            w2v_embed_matrix = w2v_model.vectors
            fasttext_embed_matrix = fasttext_model.vectors
#             embed_matrix = w2v_embed_matrix
            embed_matrix = np.concatenate(
                [w2v_embed_matrix, fasttext_embed_matrix], axis=1)
            oov_embed = np.zeros((1, embed_matrix.shape[1]))
            embed_matrix = torch.from_numpy(
                np.vstack((oov_embed, embed_matrix)))
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False
        self.gru1 = nn.GRU(embedding_dim, hidden_size_gru,
                           bidirectional=True, batch_first=True)
        self.att1 = SelfAttention(hidden_size_gru * 2, hidden_size_att)
        self.gru2 = nn.GRU(hidden_size_att, hidden_size_gru,
                           bidirectional=True, batch_first=True)
        self.att2 = SelfAttention(hidden_size_gru * 2, hidden_size_att)
        self.tdfc = nn.Linear(embedding_dim, embedding_dim)
        self.tdbn = nn.BatchNorm2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size_att, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.num_classes)
        )
        self.dropout = nn.Dropout(0.5)
        # self._init_parameters()

    def forward(self, x, label=None):
        # 64 512 200
        x = x.view(x.size(0) * self.num_words, -1).contiguous()
        x = self.dropout(self.embedding(x))
        x = self.tdfc(x).unsqueeze(1)
        x = self.tdbn(x).squeeze(1)
        x, _ = self.gru1(x)
        x = self.att1(x)
        x = x.view(x.size(0) // self.num_words,
                   self.num_words, -1).contiguous()
        x, _ = self.gru2(x)
        x = self.att2(x)
        out = self.dropout(self.fc(x))
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out


class Caps_Layer(nn.Module):  # 胶囊层
    def __init__(self,args, input_dim_capsule, num_capsule=5, dim_capsule=5,
                 routings=4, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)
        self.T_epsilon = 1e-7
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = 4
        self.kernel_size = kernel_size  # 暂时没用到
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(nn.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                nn.randn(args.batch_size, input_dim_capsule, self.num_capsule * self.dim_capsule))  # 64即batch_size

    def forward(self, x):

        if self.share_weights:
            # [16, 100, 256]-->[16, 100, 25]
            u_hat_vecs = nn.matmul(x, self.W)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)  # 输入是100维度
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,  # [16, 100, 25] --> [16, 100, 5, 5]
                                      self.num_capsule, self.dim_capsule))
        # 交换维度，即转成(batch_size,num_capsule,input_num_capsule,dim_capsule)  [16, 5, 100, 5]
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)
        # (batch_size,num_capsule,input_num_capsule)[16, 5, 100]
        b = nn.zeros_like(u_hat_vecs[:, :, :, 0])

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            # batch matrix multiplication
            outputs = self.activation(nn.einsum(
                'bij,bijk->bik', (c, u_hat_vecs)))
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                # batch matrix multiplication
                b = nn.einsum('bik,bijk->bij', (outputs, u_hat_vecs))
        return outputs  # (batch_size, num_capsule, dim_capsule)[16, 5, 5]

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = nn.sqrt(s_squared_norm + self.T_epsilon)
        return x / scale


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = nn.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(nn.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = nn.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = nn.tanh(eij)
        a = nn.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / nn.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * nn.unsqueeze(a, -1)  # [16, 100, 256]
        return nn.sum(weighted_input, 1)

class CapsuleNet(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, embeddings=None, num_capsule=5, dim_capsule=5,):
        super(CapsuleNet, self).__init__()
        self.num_classes = 35
        fc_layer = 256
        hidden_size = 128
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embeddings:
            w2v_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/w2v.model").wv
            fasttext_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/fasttext.model").wv
            w2v_embed_matrix = w2v_model.vectors
            fasttext_embed_matrix = fasttext_model.vectors
#             embed_matrix = w2v_embed_matrix
            embed_matrix = np.concatenate(
                [w2v_embed_matrix, fasttext_embed_matrix], axis=1)
            oov_embed = np.zeros((1, embed_matrix.shape[1]))
            embed_matrix = nn.from_numpy(
                np.vstack((oov_embed, embed_matrix)))
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.GRU(embedding_dim, hidden_size, 2,
                           bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, 2,
                          bidirectional=True, batch_first=True)
        self.tdbn = nn.BatchNorm2d(1)
        self.lstm_attention = Attention(hidden_size * 2, args.MAX_LEN)
        self.gru_attention = Attention(hidden_size * 2, args.MAX_LEN)
        self.bn = nn.BatchNorm1d(fc_layer)
        self.linear = nn.Linear(hidden_size*8+1, fc_layer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(fc_layer, self.num_classes)
        self.lincaps = nn.Linear(num_capsule * dim_capsule, 1)
        self.caps_layer = Caps_Layer(args,hidden_size*2)

    def forward(self, x, label=None):

        #         Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
        h_embedding = self.embedding(x)
        h_embedding = nn.squeeze(
            self.embedding_dropout(nn.unsqueeze(h_embedding, 0)))
        h_embedding = self.tdbn(h_embedding.unsqueeze(1)).squeeze(1)
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        ##Capsule Layer
        content3 = self.caps_layer(h_gru)
        content3 = self.dropout(content3)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)
        content3 = self.relu(self.lincaps(content3))

        ##Attention Layer
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = nn.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = nn.max(h_gru, 1)

        conc = nn.cat((h_lstm_atten, h_gru_atten,
                         content3, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.bn(conc)
        out = self.dropout(self.output(conc))
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out


class TextRCNN(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, embeddings=None):
        super(TextRCNN, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.4
        self.learning_rate = 5e-4
        self.freeze = True  # 训练过程中是否冻结对词向量的更新
        self.seq_len = 100
        self.num_classes = 35
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # 字向量维度
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if embeddings:
            w2v_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/w2v.model").wv
            fasttext_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/fasttext.model"
            ).wv
            w2v_embed_matrix = w2v_model.vectors
            fasttext_embed_matrix = fasttext_model.vectors
            #             embed_matrix = w2v_embed_matrix
            embed_matrix = np.concatenate(
                [w2v_embed_matrix, fasttext_embed_matrix], axis=1
            )
            oov_embed = np.zeros((1, embed_matrix.shape[1]))
            embed_matrix = torch.from_numpy(
                np.vstack((oov_embed, embed_matrix)))
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False

        self.spatial_dropout = SpatialDropout(drop_prob=0.3)
        self.lstm = nn.GRU(
            input_size=self.embedding_dim, hidden_size=self.hidden_size,
            num_layers=self.num_layers, bidirectional=True,
            batch_first=True, dropout=self.dropout
        )
        self.maxpool = nn.MaxPool1d(self.seq_len)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embedding_dim, 35)
        # self._init_parameters()

    def forward(self, x, label=None):
        embed = self.embedding(x)
        spatial_embed = self.spatial_dropout(embed)
        out, _ = self.lstm(spatial_embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze(-1)
        out = self.fc(out)
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out



class TextRCNNAttn(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, embeddings=None):
        super(TextRCNNAttn, self).__init__()
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.num_layers = 2
        self.dropout = 0.4
        self.learning_rate = 5e-4
        self.freeze = True  # 训练过程中是否冻结对词向量的更新
        self.seq_len = 100
        self.num_classes = 35
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # 字向量维度
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if embeddings:
            w2v_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/w2v.model").wv
            fasttext_model = gensim.models.word2vec.Word2Vec.load(
                "./embedding/fasttext.model"
            ).wv
            w2v_embed_matrix = w2v_model.vectors
            fasttext_embed_matrix = fasttext_model.vectors
            #             embed_matrix = w2v_embed_matrix
            embed_matrix = np.concatenate(
                [w2v_embed_matrix, fasttext_embed_matrix], axis=1
            )
            oov_embed = np.zeros((1, embed_matrix.shape[1]))
            embed_matrix = torch.from_numpy(
                np.vstack((oov_embed, embed_matrix)))
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False

        self.spatial_dropout = SpatialDropout(drop_prob=0.3)
        self.lstm = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size1,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout,
        )
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.hidden_size1 * 2))
        self.fc1 = nn.Linear(
            self.hidden_size1 * 2 + self.embedding_dim, self.hidden_size2
        )
        self.maxpool = nn.MaxPool1d(self.seq_len)
        self.fc2 = nn.Linear(self.hidden_size2, self.num_classes)
        # self._init_parameters()

    def forward(self, x, label=None):
        embed = self.embedding(x)
        spatial_embed = self.spatial_dropout(embed)
        H, _ = self.lstm(spatial_embed)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze(-1)
        out = self.fc1(out)
        out = self.fc2(out)
        if label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1, self.num_classes).float(),
                            label.view(-1, self.num_classes).float())
            return loss
        else:
            return out

