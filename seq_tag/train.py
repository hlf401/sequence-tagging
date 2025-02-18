import logging
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .data_utils import NerDataset, get_idx2tag, load_checkpoint, save_checkpoint
from .model import BertBilstmCrf
from . import metric

here = os.path.dirname(os.path.abspath(__file__))

# 使用测试数据集进行模型训练
# 使用验证数据集进行评价
def train(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    train_file = hparams.train_file
    validation_file = hparams.validation_file
    log_dir = hparams.log_dir
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    checkpoint_file = hparams.checkpoint_file

    max_len = hparams.max_len
    train_batch_size = hparams.train_batch_size
    validation_batch_size = hparams.validation_batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay

    print("The training hparams are:")
    print(hparams)
    '''
    The training hparams are:
    Namespace(
        checkpoint_file='D:\\code_new\\sequence-tagging\\seq_tag\\../saved_models\\checkpoint.json', 
        device='cuda', 
        dropout=0.1, 
        embedding_dim=768, 
        epochs=10, 
        learning_rate=1e-05, 
        log_dir='D:\\code_new\\sequence-tagging\\seq_tag\\../saved_models\\runs', 
        max_len=128, model_file='D:\\code_new\\sequence-tagging\\seq_tag\\../saved_models\\model.bin', 
        output_dir='D:\\code_new\\sequence-tagging\\seq_tag\\../saved_models', 
        pretrained_model_path='D:\\code_new\\sequence-tagging\\seq_tag\\../pretrained_models/bert-base-chinese', 
        rnn_bidirectional=True, 
        rnn_hidden_dim=256, 
        rnn_num_layers=1, 
        seed=12345, 
        tagset_file='D:\\code_new\\sequence-tagging\\seq_tag\\../saved_models\\tagset.txt', 
        train_batch_size=8, 
        train_file='D:\\code_new\\sequence-tagging\\seq_tag\\../datasets/train.txt', 
        validation_batch_size=8, 
        validation_file='D:\\code_new\\sequence-tagging\\seq_tag\\../datasets/validation.txt', 
        weight_decay=0)

    '''
    # 加载数据集：train_dataset， 会对数据集进行tokenize, 预处理，和编码
    train_dataset = NerDataset(train_file, tagset_path=tagset_file,
                               pretrained_model_path=pretrained_model_path,
                               max_len=max_len, is_train=True)
    # DataLoader批量迭代数据集        
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # model
    # 从tagset_file中读出所有实体和类型(idx, tag)列表
    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    print("tagset_size is: %d" % hparams.tagset_size)
    print(idx2tag)
    '''
    tagset_size is: 21
    {0: 'B-address', 1: 'B-book', 2: 'B-company', 3: 'B-game', 4: 'B-government', 5: 'B-movie', 6: 'B-name', 7: 'B-organization', 8: 'B-position', 9: 'B-scene', 10: 'I-address', 11: 'I-book', 12: 'I-company', 13: 'I-game', 14: 'I-government', 15: 'I-movie', 16: 'I-name', 17: 'I-organization', 18: 'I-position', 19: 'I-scene', 20: 'O'}
    '''



    
    model = BertBilstmCrf(hparams).to(device)

    # load checkpoint if one exists
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_file))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    running_loss = 0.0
    # log writter
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))


    # 训练
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            #  读出一批次数据
            token_ids = sample_batched['token_ids'].to(device)
            token_type_ids = sample_batched['token_type_ids'].to(device)
            attention_mask = sample_batched['attention_mask'].to(device)
            
            tag_ids = sample_batched['tag_ids'].to(device)
            model.zero_grad()

            # 调用BertBilstmCrf.forwards()预测，并返回loss 然后进行参数调整
            loss = model(token_ids, tag_ids, token_type_ids, attention_mask)
            loss.backward()
            optimizer.step()

            # 计算错误率
            running_loss += loss.item()
            if i_batch % 10 == 9:
                writer.add_scalar('Training/training loss', running_loss / 10, epoch * len(train_loader) + i_batch)
                running_loss = 0.0

            # only debug
            # break

        # 如果有验证数据集的话，每个epoch，验证一批数据，并输出评价值
        if validation_file:
            validation_dataset = NerDataset(validation_file, tagset_path=tagset_file,
                                            pretrained_model_path=pretrained_model_path,
                                            max_len=max_len, is_train=False)
            val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
            model.eval()
            with torch.no_grad():
                tags_true_list = []
                tags_pred_list = []
                for val_i_batch, val_sample_batched in enumerate(tqdm(val_loader, desc='Validation')):

                    #  读出一批次数据
                    token_ids = val_sample_batched['token_ids'].to(device)
                    token_type_ids = val_sample_batched['token_type_ids'].to(device)
                    attention_mask = val_sample_batched['attention_mask'].to(device)

                    # 实际对应的tag_ids
                    tag_ids = val_sample_batched['tag_ids'].tolist()

                    if val_i_batch == 1:
                        logging.debug("The data read from the validition datasets:")
                        logging.debug("token_ids.shape")
                        logging.debug(token_ids.shape)
                        logging.debug("tag_ids.shape")
                        logging.debug(val_sample_batched['tag_ids'].shape)   
                        # [8, 128] 8句话，每句最多128个字
                        '''
                         The data read from the validition datasets:
                        token_ids.shape
                        torch.Size([8, 128])
                        tag_ids.shape
                        torch.Size([8, 128])
                        '''
                        
                    # 预测句子tokens对应的tags, 并decode 返回tags
                    pred_tag_ids = model.decode(input_ids=token_ids, token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)
                    if val_i_batch == 1:
                        logging.debug("The pred_tag_ids by model:")
                        logging.debug("pred_tag_ids.shape")
                        logging.debug("[%d, %d]" % (len(pred_tag_ids), len(pred_tag_ids[0]))) 
                        logging.debug("[%d, %d]" % (len(pred_tag_ids), len(pred_tag_ids[1]))) 

                    # [8, 43] 8句话，第0句话有43个字的预测值

                    '''
                    The pred_tag_ids by model:
                    pred_tag_ids.shape
                    [8, 43]
                    [8, 35]
                    '''

                    seq_ends = attention_mask.sum(dim=1)
                    true_tag_ids = [_tag_ids[:seq_ends[i]] for i, _tag_ids in enumerate(tag_ids)]
                    if val_i_batch == 1:
                        logging.debug("The true_tag_ids by attention mask:")
                        logging.debug("true_tag_ids.shape")
                        #logging.debug(np.array(true_tag_ids).shape)
                        logging.debug("[%d, %d]" % (len(true_tag_ids), len(true_tag_ids[0]))) 
                        logging.debug("[%d, %d]" % (len(true_tag_ids), len(true_tag_ids[1]))) 


                        '''
                        The true_tag_ids by attention mask:
                        true_tag_ids.shape
                        [8, 43]
                        [8, 35]
                        '''

                    # 不知道这一步啥意义，貌似转不转都一样
                    batched_tags_true = [[idx2tag[tag_id] for tag_id in _tag_ids] for _tag_ids in true_tag_ids]
                    batched_tags_pred = [[idx2tag[tag_id] for tag_id in _tag_ids] for _tag_ids in pred_tag_ids]

                    if val_i_batch == 1:

                        logging.debug("The batched_tags:")
                        logging.debug("batched_tags_true.shape")
                        logging.debug("[%d, %d]" % (len(batched_tags_true), len(batched_tags_true[0]))) 
                        logging.debug("[%d, %d]" % (len(batched_tags_true), len(batched_tags_true[1]))) 

                        logging.debug("batched_tags_pred.shape")
                        logging.debug("[%d, %d]" % (len(batched_tags_pred), len(batched_tags_pred[0]))) 
                        logging.debug("[%d, %d]" % (len(batched_tags_pred), len(batched_tags_pred[1]))) 


                        '''
                        The batched_tags:
                        batched_tags_true.shape
                        [8, 43]
                        [8, 35]
                        batched_tags_pred.shape
                        [8, 43]
                        [8, 35]

                        '''

                    # 添加当前批次的tags_true and tags_pred到总列表中
                    tags_true_list.extend(batched_tags_true)
                    tags_pred_list.extend(batched_tags_pred)



                # 输出和保存评价指标
                print(metric.classification_report(tags_true_list, tags_pred_list))
                f1 = metric.f1_score(tags_true_list, tags_pred_list)
                precision = metric.precision_score(tags_true_list, tags_pred_list)
                recall = metric.recall_score(tags_true_list, tags_pred_list)
                accuracy = metric.accuracy_score(tags_true_list, tags_pred_list)
                writer.add_scalar('Validation/f1', f1, epoch)
                writer.add_scalar('Validation/precision', precision, epoch)
                writer.add_scalar('Validation/recall', recall, epoch)
                writer.add_scalar('Validation/accuracy', accuracy, epoch)

                if checkpoint_dict.get('epoch_f1'):
                    checkpoint_dict['epoch_f1'][epoch] = f1
                else:
                    checkpoint_dict['epoch_f1'] = {epoch: f1}

                # 如果f1 比较好，则保存、更新当前训练好的模型
                if f1 > best_f1:
                    best_f1 = f1
                    checkpoint_dict['best_f1'] = best_f1
                    checkpoint_dict['best_epoch'] = epoch
                    torch.save(model.state_dict(), model_file)
                save_checkpoint(checkpoint_dict, checkpoint_file)

    writer.close()
