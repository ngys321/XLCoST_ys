import json
import pandas as pd
import csv
import collections
from collections import Counter
import itertools
from itertools import product
import pickle
import random
import os
import copy
import time
start_time = time.time() # execution time 체크용. 

augmentation_only_positive = True
transformation_list = ['no', 'delete', 'switch', 'copy'] # 'no', 'delete'

with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost-train_py.json", 'r') as f:
   xlcost_train = json.load(f)

cosqa_train_query = [inst['doc'] for inst in xlcost_train] # cosqa_train_query = [inst['NL'] for inst in NL_train]
cosqa_train_code = [inst['code'] for inst in xlcost_train] # cosqa_train_code = [inst['PL'] for inst in PL_train]
cosqa_train_label = [inst['label'] for inst in xlcost_train] # cosqa_train_label = [1 for inst in PL_train] # xlcost는 모든 label == 1
cosqa_train_idx = [inst['idx'] for inst in xlcost_train] # cosqa_train_idx = ["xlcost-train-"+str(x) for x in range(len(PL_train))]

with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_d_lexicon.pickle', 'rb') as f:
    cosqa_train_d_lexicon = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_f_lexicon.pickle', 'rb') as f:
    cosqa_train_f_lexicon = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_q_lexicon.pickle', 'rb') as f:
    cosqa_train_q_lexicon = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_query_refine.pickle', 'rb') as f:
    cosqa_train_query_refine = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_docstring_refine.pickle', 'rb') as f:
    cosqa_train_docstring_refine = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_funcname_refine.pickle', 'rb') as f:
    cosqa_train_funcname_refine = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_docstring.pickle', 'rb') as f:
    cosqa_train_docstring = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_head_tokens.pickle', 'rb') as f:
    cosqa_train_code_head_tokens = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_body_tokens.pickle', 'rb') as f:
    cosqa_train_code_body_tokens = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_funcname.pickle', 'rb') as f:
    cosqa_train_code_funcname = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_prefix_funcname.pickle', 'rb') as f:
    cosqa_train_code_prefix_funcname = pickle.load(f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_sufix_funcname.pickle', 'rb') as f:
    cosqa_train_code_sufix_funcname = pickle.load(f)

with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost_train_variable.pickle', 'rb') as f:
    cosqa_train_variable = pickle.load(f)

print("-")  # cosqa_train_q_lexicon , cosqa_train_f_lexicon , cosqa_train_d_lexicon 체크

data = copy.deepcopy(xlcost_train)

extended_data = []

def delete(index, word_list, sample_type):
    flag = False

    new_word_list = copy.deepcopy(word_list)
    new_sample = copy.deepcopy(new_word_list)

    if sample_type == 'query':
        lex = copy.deepcopy(cosqa_train_q_lexicon)
    elif sample_type == 'doc':
        lex = copy.deepcopy(cosqa_train_d_lexicon)
    else:
        lex = copy.deepcopy(cosqa_train_f_lexicon)

    lex = list(set(lex[index].split()))

    none_keyword_list = []
    for w in new_word_list:
        if w in lex:
            pass
        else:
            none_keyword_list.append(w)

    if len(none_keyword_list)==0:
        flag = False
    else:
        target = random.sample(none_keyword_list, 1)[0]

        new_sample = [item for item in new_word_list if item!=target]
        flag = True

    return new_sample, flag

def switch(index, word_list, sample_type):
    flag = False

    new_word_list = copy.deepcopy(word_list)
    new_sample = copy.deepcopy(new_word_list)

    if sample_type == 'query':
        lex = copy.deepcopy(cosqa_train_q_lexicon)
    elif sample_type == 'doc':
        lex = copy.deepcopy(cosqa_train_d_lexicon)
    else:
        lex = copy.deepcopy(cosqa_train_f_lexicon)

    lex = list(set(lex[index].split()))

    none_keyword_list = []
    for w in new_word_list:
        if w in lex:
            pass
        else:
            none_keyword_list.append(w)

    if len(none_keyword_list) < 2:
        flag = False
    else:
        if len(new_sample) < 2:
            flag = False
        else:
            target1 = random.sample(none_keyword_list, 1)[0]
            none_keyword_list.remove(target1)
            target2 = random.sample(none_keyword_list, 1)[0]

            idx1, idx2 = new_word_list.index(target1), new_word_list.index(target2)
            new_sample[idx1], new_sample[idx2] = new_word_list[idx2], new_word_list[idx1]
            flag = True
            
    return new_sample, flag

def copy_insert(index, word_list, sample_type):
    flag = False

    new_word_list = copy.deepcopy(word_list)
    new_sample = copy.deepcopy(new_word_list)

    if sample_type == 'query':
        lex = copy.deepcopy(cosqa_train_q_lexicon)
    elif sample_type == 'doc':
        lex = copy.deepcopy(cosqa_train_d_lexicon)
    else:
        lex = copy.deepcopy(cosqa_train_f_lexicon)

    lex = list(set(lex[index].split()))

    none_keyword_list = []
    for w in new_word_list:
        if w in lex:
            pass
        else:
            none_keyword_list.append(w)

    if len(none_keyword_list) == 0:
        flag = False
    else:
        target = random.sample(none_keyword_list, 1)[0]
        position = new_word_list.index(target)

        new_sample.insert(position, target)

        flag = True
            
    return new_sample, flag

def no(index, word_list, sample_type):
    new_sample = copy.deepcopy(word_list)

    flag = True

    return new_sample, flag

def var_rename(index, word_list, sample_type):
    new_word_list = copy.deepcopy(word_list)
    new_sample = []

    flag = False

    variable_name = copy.deepcopy(cosqa_train_variable[index])
    lex_vocab = list()
    ql = copy.deepcopy(cosqa_train_q_lexicon[index])
    fl = copy.deepcopy(cosqa_train_f_lexicon[index])
    dl = copy.deepcopy(cosqa_train_d_lexicon[index])
    lex_vocab.extend(ql.split())
    lex_vocab.extend(fl.split())
    lex_vocab.extend(dl.split())
    lex_vocab = list(set(lex_vocab))

    tgt = ''
    replace_word = ''

    if len(variable_name) > 0 and len(lex_vocab) > 0:
        tgt = variable_name[0][0]

        tgt_start_char = tgt[0]
        for l in lex_vocab:
            if l[0] == tgt_start_char:
                replace_word = l
        if replace_word == '':        
            replace_word = random.sample(lex_vocab, 1)[0]
        else:
            pass
    
        replace_word = replace_word.replace("'", "")
        replace_word = replace_word.lstrip()
        replace_word = replace_word.rstrip()

        for c in new_word_list:
            if c == tgt:
                new_sample.append(replace_word)
            else:
                new_sample.append(c)
        flag = True
    else:
        pass
    return new_sample, flag, tgt, replace_word

def extend_sample(index, sample, query, func_name, doc, code_body, prefix_func_name, sufix_func_name, new_code_body, c_flag, tgt, replaced_word):

    index = copy.deepcopy(index)

    new_sample = copy.deepcopy(sample)
    new_query = copy.deepcopy(query)
    new_func_name = copy.deepcopy(func_name)
    new_doc = copy.deepcopy(doc)
    code_body = copy.deepcopy(code_body)
    query_kind = new_query[1]
    func_name_kind = new_func_name[1]
    doc_kind = new_doc[1]

    new_code_body = copy.deepcopy(new_code_body)
    prefix_func_name = copy.deepcopy(prefix_func_name)
    sufix_func_name = copy.deepcopy(sufix_func_name)
    replaced_word = copy.deepcopy(replaced_word)
    tgt = copy.deepcopy(tgt)

    if query_kind == 'no' and func_name_kind == 'no' and doc_kind == 'no':
        pass
    else:
        new_sample['idx'] = new_sample['idx'] + '-query:' + query_kind + '-funcname:' + func_name_kind + '-doc:' + doc_kind + '-code:' + 'no'       
        new_sample['doc'] = ' '.join(new_query[0])
        new_code = []
        new_code.extend(prefix_func_name)
        new_code.extend(new_func_name[0])
        new_code.extend(sufix_func_name)

        new_code.extend(new_doc[0])        
        new_code.extend(code_body)
        new_sample['code'] = ' '.join(new_code)
        code_tokens = []
        code_tokens.extend(prefix_func_name)
        code_tokens.extend(new_func_name[0])
        code_tokens.extend(sufix_func_name)

        code_tokens.extend(code_body)
        new_sample['code_tokens'] = ' '.join(code_tokens)
        new_sample['docstring_tokens'] = ' '.join(new_doc[0])
        extended_data.append(new_sample) 

    if c_flag == True:
        new_sample = copy.deepcopy(sample)
        new_sample['idx'] = new_sample['idx'] + '-query:' + query_kind + '-funcname:' + func_name_kind + '-doc:' + doc_kind + '-code:' + 'var_rename'       
        new_sample['doc'] = ' '.join(new_query[0])
        new_code = []
        
        new_sufix_func_name = []
        for c in sufix_func_name:
            if c == tgt:
                new_sufix_func_name.append(replaced_word)
            else:
                new_sufix_func_name.append(c)
        sufix_func_name = new_sufix_func_name            

        new_code.extend(prefix_func_name)
        new_code.extend(new_func_name[0])
        new_code.extend(sufix_func_name)
        
        new_code.extend(new_doc[0])
        new_code.extend(new_code_body)
        new_sample['code'] = ' '.join(new_code)
        code_tokens = []

        code_tokens.extend(prefix_func_name)
        code_tokens.extend(new_func_name[0])
        code_tokens.extend(sufix_func_name)
        
        code_tokens.extend(new_code_body)
        new_sample['code_tokens'] = ' '.join(code_tokens)
        new_sample['docstring_tokens'] = ' '.join(new_doc[0])

        extended_data.append(new_sample) 
        
def augment_sample(index, sample):
    new_sample = copy.deepcopy(sample)
    index = copy.deepcopy(index)

    query = copy.deepcopy(cosqa_train_query_refine[index])
    query = query.split()

    doc = copy.deepcopy(cosqa_train_docstring_refine[index])
    doc = doc.split()

    func_name = copy.deepcopy(cosqa_train_funcname_refine[index])
    func_name = func_name.split()

    code_body = copy.deepcopy(cosqa_train_code_body_tokens[index])
    code_head = copy.deepcopy(cosqa_train_code_head_tokens[index])
    
    prefix_func_name = copy.deepcopy(cosqa_train_code_prefix_funcname[index])
    sufix_func_name = copy.deepcopy(cosqa_train_code_sufix_funcname[index])
    
    prefix_doc = '"""'
    sufix_doc = '"""'

    query_variants = []
    funcname_variants = []
    doc_variants = []

    if augmentation_only_positive: # 실행됨.
        if sample['label'] == 1 or sample['label'] == '1':
            for t in transformation_list:
                if t == 'no':
                    new_q, q_flag = no(index, query, 'query')
                    new_f, f_flag = no(index, func_name, 'func_name')
                    new_d, d_flag = no(index, doc, 'doc')
                elif t == 'delete':
                    new_q, q_flag = delete(index, query, 'query')
                    new_f, f_flag = delete(index, func_name, 'func_name')
                    new_d, d_flag = delete(index, doc, 'doc')
                elif t == 'switch':
                    new_q, q_flag = switch(index, query, 'query')
                    new_f, f_flag = switch(index, func_name, 'func_name')
                    new_d, d_flag = switch(index, doc, 'doc')
                elif t == 'copy':
                    new_q, q_flag = copy_insert(index, query, 'query')
                    new_f, f_flag = copy_insert(index, func_name, 'func_name')
                    new_d, d_flag = copy_insert(index, doc, 'doc')
                else:
                    pass

                if q_flag == True:
                    query_variants.append([new_q, t])
                if f_flag == True:
                    funcname_variants.append([new_f, t])
                if d_flag == True:
                    tmp = []
                    tmp.append(prefix_doc)
                    tmp.extend(new_d)
                    tmp.append(sufix_doc)
                    new_d = tmp
                    doc_variants.append([new_d, t])
    else: # 실행안됨
        for t in transformation_list:
            if t == 'no':
                new_q, q_flag = no(index, query, 'query')
                new_f, f_flag = no(index, func_name, 'func_name')
                new_d, d_flag = no(index, doc, 'doc')
            elif t == 'delete':
                new_q, q_flag = delete(index, query, 'query')
                new_f, f_flag = delete(index, func_name, 'func_name')
                new_d, d_flag = delete(index, doc, 'doc')
            elif t == 'switch':
                new_q, q_flag = switch(index, query, 'query')
                new_f, f_flag = switch(index, func_name, 'func_name')
                new_d, d_flag = switch(index, doc, 'doc')
            elif t == 'copy':
                new_q, q_flag = copy_insert(index, query, 'query')
                new_f, f_flag = copy_insert(index, func_name, 'func_name')
                new_d, d_flag = copy_insert(index, doc, 'doc')
            else:
                pass

            if q_flag == True:
                query_variants.append([new_q, t])
            if f_flag == True:
                funcname_variants.append([new_f, t])
            if d_flag == True:
                tmp = []
                tmp.append(prefix_doc)
                tmp.extend(new_d)
                tmp.append(sufix_doc)
                new_d = tmp
                doc_variants.append([new_d, t])

    new_code_body, c_flag, tgt, replaced_word = var_rename(index, code_body, 'code')
    all_combinations = list(product(query_variants, funcname_variants, doc_variants))
    for ac in all_combinations:
        extend_sample(index, new_sample, ac[0], ac[1], ac[2], code_body, prefix_func_name, sufix_func_name, new_code_body, c_flag, tgt, replaced_word)
    
    
for i, inst in enumerate(data):
    augment_sample(i, inst)
    print("index : "+str(i)+"\t"+"execution time : "+str(time.time() - start_time))

print("extended data num. {} ".format(len(extended_data)))

print("data num. before extension {} ".format(len(data)))
data.extend(extended_data)

print("data num. after extension {} ".format(len(data)))

file_name = "/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost-train_py.json" #"data/qa/cosqa-train.json"

if augmentation_only_positive:
    save_file_name = file_name.split('.json')[0] + '-{}-{}.json'.format('keydac', 'only_positive')
else:
    save_file_name = file_name.split('.json')[0] + '-{}-{}.json'.format('keydac', 'positive_negative')

print("data save to: {} ".format(save_file_name))
with open(save_file_name, 'w') as fp:
    json.dump(data, fp, indent=1)