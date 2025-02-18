import logging
import os
import torch
import torch.nn as nn

from transformers import BertModel

from .layers import CRF

here = os.path.dirname(os.path.abspath(__file__))

# 定义下游任务模型
class BertBilstmCrf(nn.Module):

    def __init__(self, hparams):
        super(BertBilstmCrf, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
        self.embedding_dim = hparams.embedding_dim      # 768
        self.rnn_hidden_dim = hparams.rnn_hidden_dim    # 256
        self.rnn_num_layers = hparams.rnn_num_layers    # 1
        self.rnn_bidirectional = hparams.rnn_bidirectional
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size          # 21

        # 载入BERT预训练模型
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)


        
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.rnn_hidden_dim // (2 if self.rnn_bidirectional else 1),
                            num_layers=self.rnn_num_layers, batch_first=True,
                            bidirectional=self.rnn_bidirectional)
        self.drop = nn.Dropout(self.dropout)
        # 隐藏层到最终分类
        # self.tagset_size =21 分类数目
        self.hidden2tag = nn.Linear(self.rnn_hidden_dim, self.tagset_size)
        
        self.crf = CRF(num_tags=self.tagset_size)
        

    def _get_emission_scores(self, input_ids, token_type_ids=None, attention_mask=None):

        # 输入：tokens信息
        embeds = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)[0]
        logging.debug("_get_emission_scores() after bert_model():")
        logging.debug("embeds.shape:")
        logging.debug(embeds.shape)
        '''
        _get_emission_scores() after bert_model():
        embeds.shape:
        torch.Size([8, 128, 768])

        '''
        lstm_out, _ = self.lstm(embeds)
        logging.debug("_get_emission_scores() after lstm():")
        logging.debug("lstm_out.shape:")
        logging.debug(lstm_out.shape)
        '''
        _get_emission_scores() after lstm():
        lstm_out.shape:
        torch.Size([8, 128, 256])

        '''
        
        lstm_dropout = self.drop(lstm_out)
        logging.debug("_get_emission_scores() after drop():")
        logging.debug("lstm_dropout.shape:")
        logging.debug(lstm_dropout.shape)
        '''
        _get_emission_scores() after drop():
        lstm_dropout.shape:
        torch.Size([8, 128, 256])

        '''

        emissions = self.hidden2tag(lstm_dropout)
        logging.debug("_get_emission_scores() after hidden2tag():")
        logging.debug("emissions.shape:")
        logging.debug(emissions.shape)   

        '''
        _get_emission_scores() after hidden2tag():
        emissions.shape:
        torch.Size([8, 128, 21])

        '''
        # 预测出tokens对应的tags
        return emissions

    # 使用模型预测，并返回 loss   
    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    # 预测句子tokens对应的tags, 并decode 返回tags
    def decode(self, input_ids, token_type_ids=None, attention_mask=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask)
        tags = self.crf.decode(emissions, mask=attention_mask.byte())
        return tags
