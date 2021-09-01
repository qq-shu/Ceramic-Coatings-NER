import torch
import torch.nn as nn
import numpy as np
from crf import CRF
from charbilstm import CharBiLSTM
from BiMogLSTM import BiMogLSTM




class NamedEntityRecog(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, word_hidden_dim, alphabet_size, char_embedding_dim, char_hidden_dim,
                 character_embedding,feature_extractor, tag_num, dropout,mog,charmog ,pretrain_embed=None, use_char=False, use_crf=False, use_gpu=False):
        super(NamedEntityRecog, self).__init__()
        self.mog = mog
        self.charmog = charmog
        self.use_crf = use_crf
        self.use_char = use_char
        self.drop = nn.Dropout(dropout)
        self.input_dim = word_embed_dim
        self.feature_extractor = feature_extractor
        self.character_embedding = character_embedding
        self.embeds = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))
        else:
            self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(vocab_size, word_embed_dim)))

        if self.use_char:
                self.input_dim += char_hidden_dim*2
                self.char_feature = CharBiLSTM(alphabet_size, char_embedding_dim, char_hidden_dim, dropout)
        if self.feature_extractor =='BiMogLSTM':
            self.bimoglstm=BiMogLSTM(self.input_dim, word_hidden_dim, mog)

        if self.use_crf:
                self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num + 2)
                self.crf = CRF(tag_num, use_gpu)
        else:
                self.hidden2tag = nn.Linear(word_hidden_dim * 2, tag_num)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        if self.use_char:
            char_features = self.char_feature(char_inputs).contiguous().view(batch_size, seq_len, -1)
            word_list.append(char_features)

        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)


        if self.feature_extractor == 'BiMogLSTM':
            bimoglstm_out, (hn, cn) = self.bimoglstm(word_represents)
            feature_out = self.drop(bimoglstm_out)

        feature_out = self.hidden2tag(feature_out)


        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(feature_out, mask, batch_label)
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            total_loss = loss_function(feature_out, batch_label.contiguous().view(batch_size * seq_len))
        return total_loss


    def forward(self, word_inputs, word_seq_lengths, char_inputs, batch_label, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        word_embeding = self.embeds(word_inputs)
        word_list = [word_embeding]
        if self.use_char:
            char_features = self.char_feature(char_inputs).contiguous().view(batch_size, seq_len, -1)
            word_list.append(char_features)

        word_embeding = torch.cat(word_list, 2)
        word_represents = self.drop(word_embeding)

        if self.feature_extractor == 'BiMogLSTM':
            bimoglstm_out, (hn, cn) = self.bimoglstm(word_represents)
            feature_out = self.drop(bimoglstm_out)

        feature_out = self.hidden2tag(feature_out)


        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(feature_out, mask)
        else:
            feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(feature_out, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq
