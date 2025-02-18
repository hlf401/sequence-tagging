import logging
import re
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

here = os.path.dirname(os.path.abspath(__file__))

# input_file: 如训练集datasets/train.txt
# 返回值：
#   tokens_list，tags_list：训练集中所有句子的tokens 和对应的分类tags
#   tagset: 所有的分类（不包含重复的）
def read_data(input_file):
    tokens_list = []
    tags_list = []  # 按序排列，有重复，对应tokens_list的tags
    
    tagset = set()  # 没有重复的
    with open(input_file, 'r', encoding='utf-8') as f_in:
        tokens = []
        tags = []
        for line in f_in:
            line = line.strip()
            # line 如：银	I-company
         
            if line:
                #提取token 如 '银'，提取tag 如 'I-company'
                token, tag = re.split(r'\s+', line, maxsplit=1)
                
                tokens.append(token)
                tags.append(tag)

                
                tagset.add(tag)
            else: # 遇见空行，则认为是下一句，重新记tokens and tags
                if tokens and tags:
                    tokens_list.append(tokens)
                    tags_list.append(tags)
                tokens = []
                tags = []
    tagset = list(tagset)
    tagset.sort()
    return tokens_list, tags_list, tagset


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict

# 定义数据集
# __getitem__
#   返回idx句子对应的所有已经编码的数据
#       包含句子信息 int，和对应的分类tags int
class NerDataset(Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128, is_train=False):
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path
        
        # 加载tokenlizer
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.max_len = max_len
        self.is_train = is_train


        # 读取训练集，如datasets/train.txt, 输出token list and tag list
        if is_train:
            self.tokens_list, self.tags_list, self.tagset = read_data(data_file_path)
            # 保存训练集中所有可能的分类tags  到saved_models\\tagset.txt
            save_tagset(self.tagset, self.tagset_path)
        else:
            self.tokens_list, self.tags_list, _ = read_data(data_file_path)

        # 读取saved_models\\tagset.txt，输出(tag, idx) 列表
        self.tag2idx = get_tag2idx(self.tagset_path)

    def __len__(self):
        return len(self.tags_list)

    # 获取某一句idx的数据：
    # 返回idx对应的所有已经编码的数据
    # 包含句子信息 int，和对应的分类tags int
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_tags = self.tags_list[idx]
        logging.debug("idx:")
        logging.debug(idx)
        if idx == 1137:
            logging.debug("original tokens and tags:")
            logging.debug("sample_tokens:")
            logging.debug(sample_tokens)
            logging.debug("sample_tags:")
            logging.debug(sample_tags)
        '''
        original tokens and tags:
        sample_tokens:
        ['也', '可', '以', '在', '商', '场', '刷', '卡', '消', '费', '。', '而', '由', '于', '是', '银', '联', '卡', '，', '因', '此', '持', '有', '韩', '亚', '银', '行', '借', '记', '卡', '在', '韩', '国', '消', '费', '时', '，']
        sample_tags:
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        '''
        # idx句对应的编码token
        encoded = self.tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        if idx == 1137:
            logging.debug("tokens after encode_plus:")
            logging.debug("sample_token_ids:")
            logging.debug(sample_token_ids)
            logging.debug("sample_token_type_ids:")
            logging.debug(sample_token_type_ids)
            logging.debug("sample_attention_mask:")
            logging.debug(sample_attention_mask)
        '''
        
        tokens after encode_plus:
        sample_token_ids:
        [101, 738, 1377, 809, 1762, 1555, 1767, 1170, 1305, 3867, 6589, 511, 5445, 4507, 754, 3221, 7213, 5468, 1305, 8024, 1728, 3634, 2898, 3300, 7506, 762, 7213, 6121, 955, 6381, 1305, 1762, 7506, 1744, 3867, 6589, 3198, 8024, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sample_token_type_ids:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sample_attention_mask:
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        '''
        
        # handing: 最后2个去除，前面添加一个'O'，最后全部补'O'，总长self.max_len
        # 书上应该讲过原因
        sample_tags = sample_tags[:self.max_len - 2]
        sample_tags = ['O'] + sample_tags + ['O'] * (self.max_len - len(sample_tags) - 1)
        if idx == 1137:
            logging.debug("tags after handling:")
            logging.debug("sample_tags:")
            logging.debug(sample_tags)

        '''
        tags after handling:
        sample_tags:
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        '''
        sample_tag_ids = [self.tag2idx[tag] for tag in sample_tags]
        if idx == 1137:
            logging.debug("tags after encode tags:")
            logging.debug("sample_tag_ids:")
            logging.debug(sample_tag_ids)
        '''
        tags after encode tags:
        sample_tag_ids:
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2, 12, 12, 12, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

        '''


        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'tag_ids': torch.tensor(sample_tag_ids)
        }
        return sample
