import os
import torch

from transformers import BertTokenizer

from .data_utils import get_idx2tag
from .model import BertBilstmCrf
from .metric import get_entities

here = os.path.dirname(os.path.abspath(__file__))


def predict(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file

    # 保存的训练好的模型文件model.bin
    model_file = hparams.model_file

    # [id, tag] 列表
    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = BertBilstmCrf(hparams).to(device)
    # 载入训练好的模型
    model.load_state_dict(torch.load(model_file))

    # 进入预测模式
    model.eval()

    # load tokenlizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    while True:
        # 输入句子
        text = input("输入中文句子：")

        # 对tokenize
        tokens = tokenizer.tokenize(text)
        # 预处理 训练是貌似没有这么处理？
        new_tokens = ['[CLS]'] + tokens + ['SEP']

        # 对tokens进行编码
        encoded = tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        token_type_ids = encoded['token_type_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            # 根据tokens预测tags_id
            tag_ids = model.decode(input_ids, token_type_ids, attention_mask)[0]
        # tags_id转成tags
        tags = [idx2tag[tag_id] for tag_id in tag_ids]


        print(list(zip(new_tokens, tags)))
        '''
          输入中文句子：侯凌峰和牛蔚露
        [('[CLS]', 'O'), ('侯', 'B-name'), ('凌', 'I-name'), ('峰', 'I-name'), ('和', 'O'), ('牛', 'B-name'), ('蔚', 'I-name'), ('露', 'I-name'), ('SEP', 'O')]
        name 侯凌峰
        name 牛蔚露      
        '''

        # 把类型相同的字符，组合形成识别出来的实体名
        chunks = get_entities(tags)
        print("entry chunks:")
        print(chunks)
        '''
        [('name', 1, 3), ('name', 5, 7)]
        '''
        
        for chunk_type, chunk_start, chunk_end in chunks:
            print(chunk_type, ''.join(new_tokens[chunk_start: chunk_end + 1]))
        '''
        name 侯凌峰
        name 牛蔚露
        '''
