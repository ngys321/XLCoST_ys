# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

# 터미널에서 실행 : 
# program synthesis train
# python run.py --do_train --do_eval --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --train_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.txt,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.py --dev_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/val-Python-desc-tok.txt,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/val-Python-desc-tok.py --output_dir /home/ysnamgoong42/ws/XLCoST/codet5_nl_pl_program/desc-Python --max_source_length 400 --max_target_length 400 --num_train_epochs 10 --train_steps 5000 --eval_steps 2500 --train_batch_size 16 --eval_batch_size 16 --beam_size 5 --learning_rate 5e-5

# program synthesis test
# python run.py --do_test --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --load_model_path /home/ysnamgoong42/ws/XLCoST/codet5_nl_pl_program/desc-Python/checkpoint-best-ppl/pytorch_model.bin --test_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/test-Python-desc-tok.txt,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/test-Python-desc-tok.py --output_dir /home/ysnamgoong42/ws/XLCoST/codet5_nl_pl_program/desc-Python --max_source_length 400 --max_target_length 400 --eval_batch_size 16 --beam_size 5

# program summarization train
# python run.py --do_train --do_eval --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --train_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/train-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/train-Python-desc-tok.txt --dev_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/val-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/val-Python-desc-tok.txt --output_dir /home/ysnamgoong42/ws/XLCoST/codet5_pl_nl_program/Python-desc --max_source_length 400 --max_target_length 100 --num_train_epochs 10 --train_steps 5000 --eval_steps 2500 --train_batch_size 16 --eval_batch_size 16 --beam_size 5 --learning_rate 5e-5

# program summarization test
# python run.py --do_test --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --load_model_path /home/ysnamgoong42/ws/XLCoST/codet5_pl_nl_program/Python-desc/checkpoint-best-ppl/pytorch_model.bin --test_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/test-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/test-Python-desc-tok.txt --output_dir /home/ysnamgoong42/ws/XLCoST/codet5_pl_nl_program/Python-desc --max_source_length 400 --max_target_length 100 --beam_size 5 --eval_batch_size 16

# 지누의 실패작 : 데이터센터 srun 켠채로 vscode debugger 사용하기 위한 방법 찾다가 잘 안됨.
# python -m debugpy --listen 127.0.0.1:5678 run.py --do_train --do_eval --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --train_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/train-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/train-Python-desc-tok.txt --dev_filename /home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/val-Python-desc-tok.py,/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc/Python-desc/val-Python-desc-tok.txt --output_dir /home/ysnamgoong42/ws/XLCoST/codet5_pl_nl_program/Python-desc --max_source_length 400 --max_target_length 100 --num_train_epochs 10 --train_steps 5000 --eval_steps 2500 --train_batch_size 16 --eval_batch_size 16 --beam_size 5 --learning_rate 5e-5
from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import (BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer,
                         PLBartModel)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
                 'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
                 'unixcoder': (RobertaConfig,RobertaModel,RobertaTokenizer)}

# 모델 코드 위치
# T5ForConditionalGeneration : /home/ysnamgoong42/miniconda3/envs/xlcost/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py
# RobertaModel : /home/ysnamgoong42/miniconda3/envs/xlcost/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py
#
## 모델 코드 위치 출력
#import inspect
#print("T5ForConditionalGeneration's code file dir : ",inspect.getfile(T5ForConditionalGeneration)) #/home/ysnamgoong42/miniconda3/envs/xlcost/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py
#print("RobertaModel's code file dir: ",inspect.getfile(RobertaModel)) #/home/ysnamgoong42/miniconda3/envs/xlcost/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

# def read_examples(filename):
#     """Read examples from filename."""
#     examples=[]
#     with open(filename,encoding="utf-8") as f:
#         for idx,js in enumerate(json.load(f)):
#             source=' '.join(js['old_comment_tokens'])
#             target=' '.join(js['new_comment_tokens'])      
#             examples.append(
#                 Example(
#                         idx = idx,
#                         source=source,
#                         target=target,
#                         ) 
#             )
#     return examples
def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    assert len(filename.split(','))==2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                examples.append(
                Example(
                        idx = idx,
                        source=line1.strip(),
                        target=line2.strip(),
                        ) 
                )
                idx+=1
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        


def convert_examples_to_features(examples, tokenizer, args,stage=None): # tokenizer 기능, data 를 feature로 변환
    features = []
    cls_token = None
    sep_token = None
    if tokenizer.cls_token and tokenizer.sep_token:
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
    else:
        cls_token = tokenizer.bos_token
        sep_token = tokenizer.eos_token
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        # example.source : (예시) 'Maximum Prefix Sum possible by merging two given arrays | Python3 implementation of the above approach ; Stores the maximum prefix sum of the array A [ ] ; Traverse the array A [ ] ; Stores the maximum prefix sum of the array B [ ] ; Traverse the array B [ ] ; Driver code'
        # args.max_source_length : 400              * pair_data_tok_full_desc_comment/train-Python-desc-tok.txt 에서 몇개 샘플 토큰수 세어보니, 150~300개 정도 , 
        # source_tokens : ['Maximum', 'ĠPrefix', 'ĠSum', 'Ġpossible', 'Ġby', 'Ġmerging', 'Ġtwo', 'Ġgiven', 'Ġarrays', 'Ġ|', 'ĠPython', '3', 'Ġimplementation', 'Ġof', ...] 62개 (왜 62개 인지는 BPE 알고리즘 검색ㄱ)??
        source_tokens =[cls_token]+source_tokens+[sep_token]
        # cls_token : '<s>'
        # sep_token : '</s>'
        # source_tokens : ['<s>', 'Maximum', 'ĠPrefix', 'ĠSum', 'Ġpossible', 'Ġby', 'Ġmerging', 'Ġtwo', 'Ġgiven', 'Ġarrays', 'Ġ|', 'ĠPython', '3', 'Ġimplementation', 'Ġof', ..., '</s>',] 64개
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        # source_ids : [1, 13528, 10139, 9352, 3323, 635, 17256, 2795, 864, 5352, 571, 6600, 23, 4471, ...] 64개
        source_mask = [1] * (len(source_tokens))
        # source_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 64개
        padding_length = args.max_source_length - len(source_ids)
        # padding_length : 336 (=400-64)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        # tokenizer.pad_token_id : 0
        # source_ids : [1, 13528, 10139, 9352, 3323, 635, 17256, 2795, 864, 5352, 571, 6600, 23, 4471, ...] 64개 + [0,0,0,...] 336개
        source_mask+=[0]*padding_length
        # source_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 64개 + [0,0,0,...] 336개
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
            # example.target : (예시) 'def maxPresum ( a , b ) : NEW_LINE INDENT X = max ( a [ 0 ] , 0 ) NEW_LINE for i in range ( 1 , len ( a ) ) : NEW_LINE INDENT a [ i ] += a [ i - 1 ] NEW_LINE X = max ( X , a [ i ] ) NEW_LINE DEDENT Y = max ( b [ 0 ] , 0 ) NEW_LINE for i in range ( 1 , len ( b ) ) : NEW_LINE INDENT b [ i ] += b [ i - 1 ] NEW_LINE Y = max ( Y , b [ i ] ) NEW_LINE DEDENT return X + Y NEW_LINE DEDENT A = [ 2 , - 1 , 4 , - 5 ] NEW_LINE B = [ 4 , - 3 , 12 , 4 , - 3 ] NEW_LINE print ( maxPresum ( A , B ) ) NEW_LINE'
            # args.max_target_length : 400           * pair_data_tok_full_dexk_comment/train-Python-desc-tok.py 에서 몇개 샘플 토큰수 세어보니, 100~395개 정도
            # target_tokens : ['def', 'Ġmax', 'Pres', 'um', 'Ġ(', 'Ġa', 'Ġ,', 'Ġb', 'Ġ)', 'Ġ:', 'ĠNEW', '_', 'LINE', 'ĠINDENT', ...] 193개 (왜 193개 인지는 BPE 알고리즘 검색ㄱ)
        target_tokens = [cls_token]+target_tokens+[sep_token]
        # cls_token : '<s>'
        # sep_token : '</s>'
        # target_tokens : ['<s>', 'def', 'Ġmax', 'Pres', 'um', 'Ġ(', 'Ġa', 'Ġ,', 'Ġb', 'Ġ)', 'Ġ:', 'ĠNEW', '_', 'LINE', ..., '</s>'] 195개
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        # target_ids : [1, 536, 943, 12236, 379, 261, 279, 269, 324, 262, 294, 12887, 67, 5997, ...] 195개
        target_mask = [1] *len(target_ids)
        # target_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 195개
        padding_length = args.max_target_length - len(target_ids)
        # padding_length : 205
        target_ids+=[tokenizer.pad_token_id]*padding_length
        # tokenizer.pad_token_id : 0
        # target_ids : [1, 536, 943, 12236, 379, 261, 279, 269, 324, 262, 294, 12887, 67, 5997, ...] 195개 + [0,0,0,...] 205개
        target_mask+=[0]*padding_length   
        # target_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 195개 + [0,0,0,...] 205개


        # 참고 : test 할땐, 
        # target_tokens : ['<s>', 'None', '</s>']
        # target_ids : [1, 7036, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]


        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index, # example_index : (예시) 0
                 source_ids,    # source_ids : [1, 13528, 10139, 9352, 3323, 635, 17256, 2795, 864, 5352, 571, 6600, 23, 4471, ...] 64개 + [0,0,0,...] 336개
                 target_ids,    # target_ids : [1, 536, 943, 12236, 379, 261, 279, 269, 324, 262, 294, 12887, 67, 5997, ...] 195개 + [0,0,0,...] 205개
                 source_mask,   # source_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 64개 + [0,0,0,...] 336개
                 target_mask,   # target_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 195개 + [0,0,0,...] 205개
            )
        )
    return features # <example_index, source_ids, target_ids, source_mask, target_mask 모두 attribute으로 갖는 InputFeatures 클래스의 인스턴스> 를 담은 리스트


def _truncate_seq_pair(tokens_a, tokens_b,tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)+len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a)>=len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b)>=len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   # local_rank 를 -1 이 아닌 값으로 줘야, DDP 작동하는듯
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--probing_case", type=int, default=0,
                        help="Probing variable, for testing different code cases")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1 #???
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    
    args.device = device
    # Set seed
   
    set_seed(args)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
      
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config 클래스, model 클래스, tokenizer 클래스
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,do_lower_case=args.do_lower_case)        # --config_name : microsoft/unixcoder-base-nine
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path) # --config_name : Salesforce/codet5-base : https://huggingface.co/Salesforce/codet5-base/blob/main/config.json
    # config : T5Config 클래스의 인스턴스
    # args.tokenizer_name : 'Salesforce/codet5-base'
    # args.do_lower_case : False
    # tokenizer : RobertaTokenizer 클래스의 인스턴스 ###################################################### tokenizer 내부적으로 어떻게 작동??

    #budild model
    if args.model_type == 'roberta':
        encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                      beam_size=args.beam_size,max_length=args.max_target_length,
                      sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    else: # codet5 는 이거 실행
        model = model_class.from_pretrained(args.model_name_or_path)    # model : T5ForConditionalGeneration 클래스의 인스턴스
    

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)




    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        # train_examples : Example 클래스의 인스턴스를 담은 리스트. Example 클래스의 인스턴스는 idx, source, target를 attribute 으로 가짐.
        #                   idx : 0
        #                   source : 'Maximum Prefix Sum possible by merging two given arrays | Python3 implementation of the above approach ; Stores the maximum prefix sum of the array A [ ] ; Traverse the array A [ ] ; Stores the maximum prefix sum of the array B [ ] ; Traverse the array B [ ] ; Driver code'
        #                   target : 'def maxPresum ( a , b ) : NEW_LINE INDENT X = max ( a [ 0 ] , 0 ) NEW_LINE for i in range ( 1 , len ( a ) ) : NEW_LINE INDENT a [ i ] += a [ i - 1 ] NEW_LINE X = max ( X , a [ i ] ) NEW_LINE DEDENT Y = max ( b [ 0 ] , 0 ) NEW_LINE for i in range ( 1 , len ( b ) ) : NEW_LINE INDENT b [ i ] += b ...'
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        # train_features : <example_index, source_ids, target_ids, source_mask, target_mask 모두 attribute으로 갖는 InputFeatures 클래스의 인스턴스> 를 담은 리스트

        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        # f.source_ids : (예시) <[1, 13528, 10139, 9352, 3323, 635, 17256, 2795, 864, 5352, 571, 6600, 23, 4471, ...] 64개 + [0,0,0,...] 336개> 
        # all_source_ids : <[1, 13528, 10139, 9352, 3323, 635, 17256, 2795, 864, 5352, 571, 6600, 23, 4471, ...] 64개 + [0,0,0,...] 336개> 들 9263개(=training data 갯수) 모은것을 tensor 로
        # all_source_ids : shape : torch.Size([9263, 400])
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        # f.source_mask :(예시) <[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 64개 + [0,0,0,...] 336개> 
        # all_source_mask : <[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 64개 + [0,0,0,...] 336개> 들 9263개(=training data 갯수) 모은것을 tensor 로
        # all_source_mask : shape : torch.Size([9263, 400])
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        # f.target_ids : (예시) <[1, 536, 943, 12236, 379, 261, 279, 269, 324, 262, 294, 12887, 67, 5997, ...] 195개 + [0,0,0,...] 205개>
        # all_target_ids : <[1, 536, 943, 12236, 379, 261, 279, 269, 324, 262, 294, 12887, 67, 5997, ...] 195개 + [0,0,0,...] 205개> 들 9263개(=training data 갯수) 모은것을 tensor 로
        # all_target_ids : shape : torch.Size([9263, 400])
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)   
        # f.target_mask : (예시) <[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 195개 + [0,0,0,...] 205개>
        # all_target_mask : <[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...] 195개 + [0,0,0,...] 205개> 들 9263개(=training data 갯수) 모은것을 tensor 로
        # all_target_mask : shape : torch.Size([9263, 400])
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
        # TensorDataset :   Dataset wrapping tensors.
        #                   Each sample will be retrieved by indexing tensors along the first dimension. 카더라.(https://pytorch.org/docs/stable/data.html)
        #                  정확히 어케 작동하는지 모르겠음.
        # train_data : all_source_ids,all_source_mask,all_target_ids,all_target_mask 를 한번에 묶어서 train_data 로 사용?
        #              train_data = { train_data[i] | i 는 자연수, train_data[i] == (all_source_ids[i], all_source_mask[i], all_target_ids[i], all_target_mask[i]) <- tuple }
        
        

        if args.local_rank == -1: # 이거 실행됨.
            train_sampler = RandomSampler(train_data) 
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps) 
  
        # num_train_optimization_steps = args.train_steps
        # changed here to make use of num_train_epochs
        num_train_optimization_steps =  int(args.num_train_epochs * len(train_examples)//args.train_batch_size)
        eval_steps = int(num_train_optimization_steps//5)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
    
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps*args.train_batch_size//len(train_examples))
        

        model.train() #Sets the module into training mode
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        bar = range(num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader) # Out of Memory Error 발생 -> 데이터셋 사이즈를 100으로 줄여서 디버깅 시도 -> 이상한데? batch = next(train_dataloader)에서 tuple out of index error 발생 -> train_sampler = RandomSampler(train_data[:100]) # train_data 대신 train_data[:100] 로 고쳐서 디버깅 -> 성공 -> 근데 결국 model 에 들어갈때 또 메모리부족함. -> 결국 데이터센터 사용함.
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            # batch : all_source_ids 와 all_source_mask 와 all_target_ids 와 all_target_mask 각각에서 1개씩 data point 를 꺼내어 담은 리스트 (리스트 크기는 4) 즉, [torch.Size([16, 400]),torch.Size([16, 400]),torch.Size([16, 400]),torch.Size([16, 400])] 임
            # all_source_ids,all_source_mask,all_target_ids,all_target_mask 에서 꺼낸 각 data point 는 모두 shape 이 torch.Size([16, 400]) 인 torch.tensor
            # batch : [source_ids, source_mask, target_ids, target_mask] ,           각각은 torch.Size([16, 400]) 인 torch.tensor

            batch = tuple(t.to(device) for t in batch)

            source_ids,source_mask,target_ids,target_mask = batch
            # sythesis 는 이렇지만, summarization 에서는 source 와 target 의 순서가 바뀐다. 그리고 target 의 seq length 도 400이 아니라 100으로 바뀐다!!!!
            #
            # source_ids    : torch.Size([16, 400])
            #                 tensor([[    1,  1380,   526,  ...,     0,     0,     0],
            #                         [    1,    47,   571,  ...,     0,     0,     0],
            #                         [    1,  9211,   404,  ...,     0,     0,     0],
            #                         ...,
            #                         [    1,  1380,   434,  ...,     0,     0,     0],
            #                         [    1, 13528,   611,  ...,     0,     0,     0],
            #                         [    1,  1380,  1300,  ...,     0,     0,     0]], device='cuda:0')
            # source_mask   : torch.Size([16, 400])                  * encoder 에 들어가는 input(=source_ids)중 padding 토큰에는 attention 이 안걸리게 하기 위한 mask. model 안에 들어가서, encoder 의 attention_mask 로 사용됨.
            #                 tensor([[1, 1, 1,  ..., 0, 0, 0],
            #                         [1, 1, 1,  ..., 0, 0, 0],
            #                         [1, 1, 1,  ..., 0, 0, 0],
            #                         ...,
            #                         [1, 1, 1,  ..., 0, 0, 0],
            #                         [1, 1, 1,  ..., 0, 0, 0],
            #                         [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')
            # target_ids    : torch.Size([16, 400])
            #                 tensor([[    1,   536,  4908,  ...,     0,     0,     0],
            #                         [    1,   536,   417,  ...,   294, 12887,     2],
            #                         [    1,   536,  7599,  ...,     0,     0,     0],
            #                         ...,
            #                         [    1,  5666,  4233,  ...,  5997, 13868,     2],
            #                         [    1,  5666,  4233,  ...,     0,     0,     0],
            #                         [    1,   536,  1056,  ...,     0,     0,     0]], device='cuda:0')
            # target_mask   : torch.Size([16, 400])                  * decoder 에 들어가는 input(=target_ids)중 padding 토큰에는 attention 이 안걸리게 하기 위한 mask. model 안에 들어가서, decoder 의 attention_mask 로 사용됨.
            #                 tensor([[1, 1, 1,  ..., 0, 0, 0],
            #                         [1, 1, 1,  ..., 1, 1, 1],
            #                         [1, 1, 1,  ..., 0, 0, 0],
            #                         ...,
            #                         [1, 1, 1,  ..., 1, 1, 1],
            #                         [1, 1, 1,  ..., 0, 0, 0],
            #                         [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')

            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)

            else: # 이거 실행됨
                #print("모델 돌리기 시작")
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                        # model 코드 : /home/ysnamgoong42/miniconda3/envs/xlcost/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py 의 class T5ForConditionalGeneration(T5PreTrainedModel) 의 forward 부분 체크
                        # Out of Memory error 발생 -> 여기부턴 vscode debugger 사용이 안된다. 설정방법 서치가 오래걸림. -> 불편하지만, 데이터센터에서 srun 켜서, 터미널 출력보면서 디버깅함
                        # 터미널에서 실행 : 
                        # python run.py --do_train --do_eval --model_type codet5 --config_name Salesforce/codet5-base --tokenizer_name Salesforce/codet5-base --model_name_or_path Salesforce/codet5-base --train_filename /home/ngys321/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.txt,/home/ngys321/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.py --dev_filename /home/ngys321/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/val-Python-desc-tok.txt,/home/ngys321/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/val-Python-desc-tok.py --output_dir /home/ngys321/ws/XLCoST/codet5_nl_pl_program/desc-Python --max_source_length 400 --max_target_length 400 --num_train_epochs 10 --train_steps 5000 --eval_steps 2500 --train_batch_size 16 --eval_batch_size 16 --beam_size 5 --learning_rate 5e-5

                loss = outputs.loss # torch.Size([1])           l_1 ~ l_6400(=batch size 16 * max seq len 400) 를 평균내어 구한 loss
                
            
                        
            if args.n_gpu > 1:      
                loss = loss.mean() # mean() to average on multi-gpu.    # 여러 gpu 들끼리 각각 구한 loss 들을 평균냄
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)    # train_loss : 현재 step까지 누적 평균 training loss      ? 근데 왜 누적 평균을 쓰지? -> training loss 가 튀는것을 줄이기 위해(뇌피셜)
            if (global_step + 1)%100==0:
                logger.info("  step {} loss {}".format(global_step + 1,train_loss))
            nb_tr_examples += source_ids.size(0)    # source_ids.size(0) : 16 batch size
            nb_tr_steps += 1                        # nb_tr_steps : training step
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            # changed from args.eval_steps to eval_steps
            if args.do_eval and ((global_step + 1) %eval_steps == 0) and eval_flag:     # (전체 epoch 에 걸친)전체 training step 의 1/5 지점마다 evaluation 수행
                #Eval model with dev dataset
                tr_loss = 0                                  #           ?? 왜 eval 때마다 tr_loss, nb_tr_examples, nb_tr_steps 를 또 초기화시키지? 즉, 왜 eval 할때마다, training loss 의 누적평균계산을 다시 시작하지?
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else: # 실행되나?
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)      
                    eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                batch_num = 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,target_ids,target_mask = batch                  

                    with torch.no_grad():
                        if args.model_type == 'roberta':
                            _,loss,num = model(source_ids=source_ids,source_mask=source_mask,
                                               target_ids=target_ids,target_mask=target_mask)  
                        
                            
                        else:
                            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                            labels=target_ids, decoder_attention_mask=target_mask)
                            loss = outputs.loss

                    if args.n_gpu > 1:                                                                                   # 추가한 코드
                        loss = loss.mean() # mean() to average on multi-gpu.    # 여러 gpu 들끼리 각각 구한 loss 들을 평균냄    # 추가한 코드... 이거 없으면, 바로 아래의 eval_loss += loss.item() 코드가 안돌아;  코드가 엉망이네;

                    if args.model_type == 'roberta':      
                        eval_loss += loss.sum().item()
                        tokens_num += num.sum().item()
                    else:
                        eval_loss += loss.item()
                        batch_num += 1
                    
                #Pring loss of dev dataset    
                model.train()
                if args.model_type == 'roberta':
                    eval_loss = eval_loss / tokens_num
                else:
                    eval_loss = eval_loss / batch_num       # eval loss 계산

                
                result = {'eval_ppl': round(np.exp(eval_loss),5),           #  ppl(perplexity)는 그냥 exp(CELoss)라고 하네 (참고 : https://huggingface.co/docs/transformers/perplexity)
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                
                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  
                            

                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:   # 두번째 eval 때부터 실행되는듯
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else: # 첫번째 eval 때만, 실행되는듯
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
                    eval_data = TensorDataset(all_source_ids,all_source_mask)   
                    dev_dataset['dev_bleu']=eval_examples,eval_data


                
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()    # eval 진짜 시작
                p=[]
                pred_ids = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask= batch                  
                    with torch.no_grad():
                        if args.model_type == 'roberta':
                            preds = model(source_ids=source_ids, source_mask=source_mask)
#                             top_preds = [pred[0].cpu().numpy() for pred in preds]
                            
                        else:
                            #
                            if args.n_gpu > 1:  # multi-gpu 로 DDP 하기위해, preds = model.generate 대신 preds = model.module.generate 로 고침 
                                preds = model.module.generate(source_ids,                              # generate 함수는 어떻게 작동하는지는? 참고 : https://huggingface.co/docs/transformers/model_doc/t5#inference, generate() 내에 다양한 decoding methods 들이 있음. 여기선 beam search 쓰는듯. 내부에 beam_search() 가 또 있어.
                                                    attention_mask=source_mask,                        # generation 내부적인 메커니즘을 이해하려면, beam_search() 코드를 봐야할듯.
                                                    use_cache=True,
                                                    num_beams=args.beam_size,
                                                    early_stopping=False, # 如果是summarize就设为True
                                                    max_length=args.max_target_length)
                            else:
                                preds = model.generate(source_ids,                              
                                                    attention_mask=source_mask,      
                                                    use_cache=True,
                                                    num_beams=args.beam_size,
                                                    early_stopping=False, # 如果是summarize就设为True
                                                    max_length=args.max_target_length)
                            #
                            

                            #preds = model.generate(source_ids,                              # generate 함수는 어떻게 작동하는지는? 참고 : https://huggingface.co/docs/transformers/model_doc/t5#inference, generate() 내에 다양한 decoding methods 들이 있음. 여기선 beam search 쓰는듯. 내부에 beam_search() 가 또 있어.
                            #                       attention_mask=source_mask,              # generation 내부적인 메커니즘을 이해하려면, beam_search() 코드를 봐야할듯.
                            #                       use_cache=True,
                            #                       num_beams=args.beam_size,
                            #                       early_stopping=False, # 如果是summarize就设为True
                            #                       max_length=args.max_target_length)
                            top_preds = list(preds.cpu().numpy())
                            pred_ids.extend(top_preds)    # pred_ids 리스트는 그전까지의 pred_ids 리스트에 top_preds 리스트를 추가해서 이어붙임.
                            # pred_ids :                    즉, 지금까지 여러 배치의 top_preds 들을 누적시킨것
                            # [ array([0,   1,  32099,  ...]),
                            #   array([0,   1,  32099,  ...]),
                            #   ...,
                            #   array([0,   1,  32099,  ...]) ]
                        
                    if args.model_type == 'roberta':
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                    else:   # 이거 실행됨.
                        p = [tokenizer.decode(id, skip_special_tokens=True,                 # tokenizer : RobertaTokenizer 클래스의 인스턴스 ###################################################### tokenizer 내부적으로 어떻게 작동?
                                                     clean_up_tokenization_spaces=False)
                                              for id in pred_ids]
                model.train()
                predictions=[]
                accs=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):   # p: prediction
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(ref+'\n')
                        f1.write(gold.target+'\n')     
                        accs.append(ref==gold.target)       # ref : prediction word 1개 , gold.target : gt word 1개 -> 서로 일치하면, accs 에 True 1개씩 append

                dev_bleu=round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")),2) # bleu 계산
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4)))) 
                logger.info("  "+"*"*20)    
                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                ######################################################################################################### 여기까지
    if args.do_test: # test 는 eval 과 마찬가지인듯
        files=[]
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
#         print(files)
#         return
        for idx,file in enumerate(files):   # test 할 파일은 하나뿐인가봐. len(files) == 1 임
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)                                                         #test dataset size 는 887 인듯
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)      #all_source_ids.shape : torch.Size([887, 400])
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    #all_source_mask.shape : torch.Size([887, 400])
            eval_data = TensorDataset(all_source_ids,all_source_mask)
            



            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            pred_ids = []
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask= batch                                           # batch 체크
                # batch : 크기 2 인 tuple
                # batch[0] : batch size(=16) 개수만큼 source_ids 를 가져옴,  batch[0].shape : torch.Size([16, 400])
                #           tensor([[   1, 9459,  358,  ...,    0,    0,    0],
                #                   [   1,   47,  451,  ...,    0,    0,    0],
                #                   [   1, 1564,  309,  ...,    0,    0,    0],
                #                   ...,
                #                   [   1, 7037,  304,  ...,    0,    0,    0],
                #                   [   1, 7037,  304,  ...,    0,    0,    0],
                #                   [   1, 3125,  326,  ...,    0,    0,    0]], device='cuda:0')
                # batch[1] : batch size(=16) 개수만큼 source_mask 를 가져옴,  batch[1].shape : torch.Size([16, 400])
                #           tensor([[1, 1, 1,  ..., 0, 0, 0],
                #                   [1, 1, 1,  ..., 0, 0, 0],
                #                   [1, 1, 1,  ..., 0, 0, 0],
                #                   ...,
                #                   [1, 1, 1,  ..., 0, 0, 0],
                #                   [1, 1, 1,  ..., 0, 0, 0],
                #                   [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')
                
                # 여긴 돌아가냐?
                #print("$$$$$$ 여기 돌아가냐? 왜 tokenizer.decode(top_preds[0]) 이거 안나옴? $$$$$$$")
                with torch.no_grad():
                    if args.model_type == 'roberta':
                        preds = model(source_ids=source_ids, source_mask=source_mask)
                        #top_preds = [pred[0].cpu().numpy() for pred in preds]
                    else: # 실행됨
                        if args.n_gpu > 1:  # multi-gpu 로 DDP 하기위해, preds = model.generate 대신 preds = model.module.generate 로 고침 
                            preds = model.module.generate(source_ids,                              # generate 함수는 어떻게 작동하는지는? 참고 : https://huggingface.co/docs/transformers/model_doc/t5#inference, generate() 내에 다양한 decoding methods 들이 있음. 여기선 beam search 쓰는듯. 내부에 beam_search() 가 또 있어.
                                                attention_mask=source_mask,                        # generation 내부적인 메커니즘을 이해하려면, beam_search() 코드를 봐야할듯.
                                                use_cache=True,
                                                num_beams=args.beam_size,
                                                early_stopping=False, # 如果是summarize就设为True
                                                max_length=args.max_target_length)
                        else:
                            preds = model.generate(source_ids,                              
                                                attention_mask=source_mask,      
                                                use_cache=True,
                                                num_beams=args.beam_size,
                                                early_stopping=False, # 如果是summarize就设为True
                                                max_length=args.max_target_length)
                            # preds.shape : torch.Size([16, 101])
                            #               tensor([[    0,     1, 32099,     1,  1071,   667,     2,     0,     0,     0,...,],
                            #                       [    0,     1, 32099,     1,   259,    20,   254,     0,     0,     0,...,],
                            #                       ...
                            #                       [    0,     1, 32099,     1,   445,   261,   262,   288, 32098,   915,...,]], device='cuda:0')
  
                    top_preds = list(preds.cpu().numpy())   # 2차원 텐서를 2차원 리스트(?) 로 변환
                    # top_preds :                           배치 1개의 pred ids 들
                    # [ array([0,   1,  32099,  ...]),
                    #   array([0,   1,  32099,  ...]),
                    #   ...,
                    #   array([0,   1,  32099,  ...]) ]
                    
                    pred_ids.extend(top_preds)  # pred_ids 리스트는 그전까지의 pred_ids 리스트에 top_preds 리스트를 추가해서 이어붙임.
                    # pred_ids :                           즉, 지금까지 여러 배치의 top_preds 들을 누적시킨것
                    # [ array([0,   1,  32099,  ...]),
                    #   array([0,   1,  32099,  ...]),
                    #   ...,
                    #   array([0,   1,  32099,  ...]) ]

                if args.model_type == 'roberta':
                    flag = 0
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)

                        if args.probing_case == 0:    # 이게 맞음      
                            if flag == 0:
                                print("tokenizer.decode(t,:",tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)) #top_preds -> p
                                flag = 1
                    # probing for testing different code cases
                    ##############################################
                    #     if args.probing_case == 1:    # 이게 맞음      
                    #         if flag == 0:
                    #             print("tokenizer.decode(t,:",tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)) #top_preds -> p
                    #             flag = 1
                    #     if args.probing_case == 2:
                    #         if flag == 0:
                    #             print("tokenizer.decode(text,:",tokenizer.decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=False)) #top_preds -> p
                    #             flag = 1
                    # if args.probing_case == 3:
                    #     print("tokenizer.decode(text,:",tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=False)) #top_preds -> p                        
                    # if args.probing_case == 4:
                    #     print("tokenizer.decode(text,:",tokenizer.decode(p[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)) #top_preds -> p                        
                    ##############################################

                # 지금 여기 바로 아래 else 내부로 못들어가는 듯
                else:  # 실행됨
                    p = [tokenizer.decode(id, skip_special_tokens=True,                  # tokenizer : RobertaTokenizer 클래스의 인스턴스 ###################################################### tokenizer 내부적으로 어떻게 작동?
                                                 clean_up_tokenization_spaces=False)
                                          for id in pred_ids]
                    # 지금까지 여러 배치의 pred ids 들을 누적시킨 pred_ids 에서, seq prediction(=id) 을 하나씩 꺼내서, tokenizer.decode() 에 넣어서, output을 얻는다. p는 그 output들로 만든 리스트.
                    
                    print("tokenizer.decode(top_preds[0],:",tokenizer.decode(top_preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                    # 배치의 첫번째 seq prediction 인 top_preds[0] 를 tokenizer.decode() 에 넣고, 출력해보면,
                    #
                    # <after training>
                    # import math NEW_LINE def findArea ( a , b ) : NEW_LINE INDENT area = a * b NEW_LINE return area NEW_LINE DEDENT a = 5 NEW_LINE b = 5 NEW_LINE print ( " Area ▁ of ▁ a ▁ Circular ▁ Sector ▁ = ▁ " , findArea ( a , b ) ) NEW_LINE
                    # 
                    # def maxIntersections ( x , y ) : NEW_LINE INDENT circles = x * ( x + 1 ) // 2 NEW_LINE straightlines = y * ( y + 1 ) // 2 NEW_LINE return ( circles + straightlines ) NEW_LINE DEDENT x = 3 NEW_LINE y = 5 NEW_LINE print ( maxIntersections ( x , y ) ) NEW_LINE
                    #
                    # <before training>
                    #
                    # function ( ) {function ( n ) {( n) { return( n ) ; }function ( n ) {( n ) ;( n ) ;( n ) ;() ; }( n ) ;() ; }) ; }) ; }) ; }() ; }) ; }() ; }() ; }
                    # 
                    # public class{}

            model.train()
            predictions=[]
            accs=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):   
                    # p: prediction 들 리스트
                    # eval_examples : Example class 의 인스턴스들의 리스트. 인스턴스는 모두 887개.
                    #                 Example class 의 attribute : idx : ex. 0
                    #                                              source : ex. 'Minimum sum possible by removing all occurrences of any array element | Function to find ...'
                    #                                              target : ex. "def minSum ( A , N ) : NEW_LINE INDENT mp = { } NEW_LINE sum = 0 NEW_LINE for i in range ..."
                    #
                    # ref : seq prediction 1개
                    # gold : test data point(Example class 의 인스턴스) 1개
                    predictions.append(str(gold.idx)+'\t'+ref)          # 여기서 append 열심히 해놓고, 왜 나중에 predictions 는 안써??
                    f.write(ref+'\n')   # seq prediction 1개 write
                    f1.write(gold.target+'\n')  # seq target 1개 write    
                    accs.append(ref==gold.target)       # ref : seq prediction 1개   ,    gold.target : seq target 1개     ->     이 두개 seq가 서로 아예 일치하면, accs 에 True 가 append, otherwise False 가 append 됨
                                                        #                                                                        ref 랑 gold.target 비교할때마다, accs 리스트에 True 또는 False 값이 추가됨.  
                    #print("ref:",ref)                   # ref : seq prediction 1개 
                    #print("gold.target:",gold.target)   # gold.target : seq target 1개 
                    #print("type(accs):",type(accs)) # <class 'list'>
                    #print("len(accs):",len(accs)) # 1, 2, 3, ..., 887

            dev_bleu=round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file),           # bleu 계산 코드는 xlcost 자체적으로 구현함.
                                 os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)),2)      # bleu 계산방법 ?? : 
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))  # np.mean(accs) : accs 리스트 안에 True, False 들이 있음. True 는 1로, False 는 0으로 간주. np.mean(accs) 는 accs 리스트 안의 1 과 0 의 평균값을 계산. 즉, 전체 원소 갯수중 True 의 비율을 계산. 0.0 ~ 1.0
                                                                                  #                 100 을 곱하여, 백분율로 계산
            logger.info("  "+"*"*20)    



                            

                
                
if __name__ == "__main__":
    main()


