import json
import pandas as pd
import csv
import collections
from collections import Counter
import itertools
import pickle
import random
import os
import copy
import time
start_time = time.time() # execution time 체크용. 860s 걸림

#### 파일 .json 아니라 .py , .txt 인데 어케?
#    파일 두개 받도록 수정해야할듯
#with open("data/qa/cosqa-train.json", 'r') as fp: # "Supplementary Material_Code/code_qa_data_augmentation/data/qa/cosqa-train.json"
#    cosqa_train = json.load(fp)
with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost-train_py.json", 'r') as f: # "Supplementary Material_Code/code_qa_data_augmentation/data/qa/cosqa-train.json"
   xlcost_train = json.load(f)


from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/parser/my-languages.so', lang) # 'Supplementary Material_Code/code_qa_data_augmentation/parser/my-languages.so'
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
                                     
def extract_dataflow(code, parser,lang):
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg, tokens_index, index_to_code

parser = parsers['python']
lang = 'python' # lang 은 python 만 썼네?

#### cosqa_train 데이터포맷 고려해야
data = copy.deepcopy(xlcost_train)
####

xlcost_train_variable = []
for inst in data:
    #### data 데이터포맷 고려해야
    #    inst['code'] -> 수정해야
    code = inst['code']
    ####

    code_tokens, dfg, tokens_index, index_to_code = extract_dataflow(code, parser, lang)
    variable = []
    for item in dfg:
        if item[2] == 'computedFrom':
            if item[0] != '_':
                variable.append(item[0])
    variable_list = Counter(variable).most_common()
    xlcost_train_variable.append(variable_list)

#print("---") # xlcost_train_variable 체크

#### cosqa_train_variable 이름 바꿀필요는 없나 -> xlcost_train_variable 로 바꿈
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost_train_variable.pickle', 'wb') as f:
    pickle.dump(xlcost_train_variable, f)

from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
wlem = nltk.WordNetLemmatizer()

import keyword
keyword_list = keyword.kwlist
datatype_list = ['boolean', 'bool', 'true', 'false', 'int', 'float', 'complex', 'str', 'string', 'list', 'tuple', 'dictionary', 'dict', 'set', 'lists', 'sets', 'dicts', 'dictionaries', 'tuples', 'strings', 'integer', 'integers']
lang_list = ['python', 'java', 'javascript', 'ruby', 'php', 'html', 'css', 'swift']

import re

####
cosqa_train_query = [inst['doc'] for inst in xlcost_train] # cosqa_train_query = [inst['NL'] for inst in NL_train]
cosqa_train_code = [inst['code'] for inst in xlcost_train] # cosqa_train_code = [inst['PL'] for inst in PL_train]
cosqa_train_label = [inst['label'] for inst in xlcost_train] # cosqa_train_label = [1 for inst in PL_train] # xlcost는 모든 label == 1
cosqa_train_idx = [inst['idx'] for inst in xlcost_train] # cosqa_train_idx = ["xlcost-train-"+str(x) for x in range(len(PL_train))]
####
#print("---") #cosqa_train_query , cosqa_train_code , cosqa_train_label , cosqa_train_idx 체크

cosqa_train_docstring = []  # xlcost 에선 doc 에 code 전체를 넣자
cosqa_train_code_funcname = [] 
cosqa_train_code_prefix_funcname = [] 
cosqa_train_code_sufix_funcname = [] 

cosqa_train_code_head_tokens = [] 
cosqa_train_code_body_tokens = []

for code in cosqa_train_code: # xlcost 에선 doc 에 code 전체를 넣자
#     items = code.split('"""')
#     doc = items[1]
#     doc = doc.replace("\n", "")
#     doc = " ".join(doc.split())
#     doc = doc.lstrip()
#     doc = doc.rstrip()
    cosqa_train_docstring.append(code)

cosqa_train_qf_inter = []
cosqa_train_qd_inter = []
cosqa_train_df_inter = []
cosqa_train_q_lexicon = []
cosqa_train_f_lexicon = []
cosqa_train_d_lexicon = []

cosqa_train_query_refine = []
cosqa_train_docstring_refine = []
cosqa_train_funcname_refine = []

#### q, d, c -> ?
#    새로 짜기 귀찮은데, 이걸 그대로 쓸수 있을까? yes
#    NL -> q
#    PL -> d
#    PL -> c -> funcname, prefix, suffix
#
#    q, d, funcname --<term matching>-->  q_lexicon, d_lexicon, f_lexicon
#
#
count = 0
for q, d, c in zip(cosqa_train_query, cosqa_train_docstring, cosqa_train_code):
    print("index : "+str(count)+"\t"+"execution time : "+str(time.time() - start_time))
    count = count + 1

    query = q
    query = query.lower()
    query = query.replace('"', '')
    tmp = []
    for w in query.split():
        tmp.append(''.join(c for c in w if c.isalnum()))
    query = ' '.join(tmp)
 
    doc = d
    doc = doc.lower()
    # doc = doc.replace('"', '')
    # tmp = []
    # for w in doc.split():
    #     tmp.append(''.join(c for c in w if c.isalnum()))
    # doc = ' '.join(tmp)

    code = c
    code_tokens, _, _, _ = extract_dataflow(code, parser, lang)
    
    if ':' in code_tokens:
        idx = code_tokens.index(':')
        head = code_tokens[:idx+1]
        body = code_tokens[idx+1:]
    else:
        head = []
        body = code_tokens

    cosqa_train_code_head_tokens.append(head)
    cosqa_train_code_body_tokens.append(body)

    func_name = ""
    if 'def' in head:
        func_name = head[head.index('def')+1:head.index('(', head.index('def')+1 ) ] # func_name = head[head.index('def')+1:head.index('(')] 
        func_name = func_name[0]
        prefix_funcname = head[:head.index('def')+1]
        sufix_funcname = head[head.index('('):]
    cosqa_train_code_funcname.append(func_name)
    cosqa_train_code_prefix_funcname.append(prefix_funcname) # prefix_funcname 없어서 에러 예상
    cosqa_train_code_sufix_funcname.append(sufix_funcname) # sufix_funcname 없어서 에러 예상
    func_name = func_name.split('_')
    func_name = ' '.join(func_name)
    tmp = []
    for w in func_name.split():
        tmp.append(''.join(c for c in w if c.isalnum()))
    func_name = ' '.join(tmp)

    query = query.split()
    if ' ' in query:
        query.remove(' ')
    query = [q.replace(' ', '') for q in query]
    query = ' '.join(query)

    # doc = doc.split()
    # if ' ' in doc:
    #     doc.remove(' ')
    # doc = [d.replace(' ', '') for d in doc]
    # doc = ' '.join(doc)

    func_name = func_name.split()
    if ' ' in func_name:
        func_name.remove(' ')
    func_name = [f.replace(' ', '') for f in func_name]
    func_name = ' '.join(func_name)

    func_name = func_name.lower()

    cosqa_train_query_refine.append(query)
    cosqa_train_docstring_refine.append(doc)
    cosqa_train_funcname_refine.append(func_name)

    query = query.split()
    doc = doc.split()
    func_name = func_name.split()

    qf_inter = list(set(query) & set(func_name))
    qf_inter = list(set(qf_inter))
    qf_inter = ' '.join(qf_inter)
    
    qd_inter = list(set(query) & set(doc))
    qd_inter = list(set(qd_inter)) 
    qd_inter = ' '.join(qd_inter)
    
    df_inter = list(set(doc) & set(func_name))
    df_inter = list(set(df_inter)) 
    df_inter = ' '.join(df_inter)
    
    df_inter = word_tokenize(df_inter)
    df_inter = [w for w in df_inter if not w in stopwords.words() and not w in keyword_list and not w in datatype_list and not w in lang_list and len(w) > 2]
    df_inter = ' '.join(df_inter)

    qf_inter = word_tokenize(qf_inter)
    qf_inter = [w for w in qf_inter if not w in stopwords.words() and not w in keyword_list and not w in datatype_list and not w in lang_list and len(w) > 2]
    qf_inter = ' '.join(qf_inter)

    qd_inter = word_tokenize(qd_inter)
    qd_inter = [w for w in qd_inter if not w in stopwords.words() and not w in keyword_list and not w in datatype_list and not w in lang_list and len(w) > 2]
    qd_inter = ' '.join(qd_inter)

    cosqa_train_df_inter.append(df_inter)
    cosqa_train_qf_inter.append(qf_inter)
    cosqa_train_qd_inter.append(qd_inter)

    qf_inter = qf_inter.split()
    qd_inter = qd_inter.split()
    df_inter = df_inter.split()

    q_lexicon = []
    q_lexicon.extend(qf_inter)
    q_lexicon.extend(qd_inter)
    q_lexicon = list(set(q_lexicon))
    q_lexicon = ' '.join(q_lexicon)

    f_lexicon = []
    f_lexicon.extend(qf_inter)
    f_lexicon.extend(df_inter)
    f_lexicon = list(set(f_lexicon))
    f_lexicon = ' '.join(f_lexicon)

    d_lexicon = []
    d_lexicon.extend(qd_inter)
    d_lexicon.extend(df_inter)
    d_lexicon = list(set(d_lexicon))
    d_lexicon = ' '.join(d_lexicon)

    cosqa_train_q_lexicon.append(q_lexicon)
    cosqa_train_f_lexicon.append(f_lexicon)
    cosqa_train_d_lexicon.append(d_lexicon)
####
# /home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost_train_variable.pickle
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_docstring.pickle', 'wb') as f:
    pickle.dump(cosqa_train_docstring, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_head_tokens.pickle', 'wb') as f:
    pickle.dump(cosqa_train_code_head_tokens, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_body_tokens.pickle', 'wb') as f:
    pickle.dump(cosqa_train_code_body_tokens, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_funcname.pickle', 'wb') as f:
    pickle.dump(cosqa_train_code_funcname, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_prefix_funcname.pickle', 'wb') as f:
    pickle.dump(cosqa_train_code_prefix_funcname, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_code_sufix_funcname.pickle', 'wb') as f:
    pickle.dump(cosqa_train_code_sufix_funcname, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_query_refine.pickle', 'wb') as f:
    pickle.dump(cosqa_train_query_refine, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_docstring_refine.pickle', 'wb') as f:
    pickle.dump(cosqa_train_docstring_refine, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_funcname_refine.pickle', 'wb') as f:
    pickle.dump(cosqa_train_funcname_refine, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_q_lexicon.pickle', 'wb') as f:
    pickle.dump(cosqa_train_q_lexicon, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_f_lexicon.pickle', 'wb') as f:
    pickle.dump(cosqa_train_f_lexicon, f)
with open('/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/cosqa_train_d_lexicon.pickle', 'wb') as f:
    pickle.dump(cosqa_train_d_lexicon, f)