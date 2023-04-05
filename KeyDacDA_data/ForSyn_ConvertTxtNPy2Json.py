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


#   text file -> json 변환 
list1 = []
with open("/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.txt", "r") as f: # ../g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.txt
    for line in f:
        dict1 = {}
        dict1["doc"] = line
        list1.append(dict1)

with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/train-Python-desc-tok_txt.json", "w") as f:
    json.dump(list1, f, indent = 4)


#   python file -> json 변환 
list2 = []
with open("/home/ysnamgoong42/ws/XLCoST/g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.py", 'r') as f: # ../g4g/XLCoST_data/pair_data_tok_full_desc_comment/Python-desc/train-Python-desc-tok.py
    for line in f:
        dict2 = {}
        dict2["code"] = line
        list2.append(dict2)

for dict2 in list2: # NEW_LINE -> \n 으로 수정, INDENT, DETECT 제거
    token_string = dict2["code"]
    token_list = token_string.split()
    for idx in range(len(token_list)):
        if token_list[idx] == 'NEW_LINE':
            token_list[idx] = '\n'
        if token_list[idx] == 'INDENT':
            token_list[idx] = '    '
        if token_list[idx] == 'DEDENT':
            token_list[idx] = ''
    dict2["code"] = ' '.join(token_list)

with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/train-Python-desc-tok_py.json", "w") as f:
    json.dump(list2, f, indent = 4)






# train-Python-desc-tok_txt.json 과 train-Python-desc-tok_py.json 를 한 파일(xlcost-train_py.json)로 
with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/train-Python-desc-tok_txt.json", "r") as f1, open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/train-Python-desc-tok_py.json", "r") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

combined_data = []
for i, item in enumerate(data1):
    combined_item = item.copy()
    combined_item.update(data2[i])
    combined_data.append(combined_item)

with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost-train_py.json", "w") as outfile:
    json.dump(combined_data, outfile, indent = 4)



# <'idx', 'code_tokens', 'docstring_tokens', 'label' 전부 채우기>
# 'idx' : xlcost-train-0 형식으로
# 'code_tokens' : 'code' 랑 똑같이
# 'docstring_tokens' : 아무것도 안넣기
# 'label' : 모두 1 로 넣기
with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost-train_py.json", "r") as f:
    data = json.load(f)

new_data = []
for i, item in enumerate(data):
    new_item = item.copy()
    new_item["idx"] = "xlcost-train-"+str(i)
    new_item["code_tokens"] = new_item["code"]
    new_item["docstring_tokens"] = ""
    new_item["label"] = 1
    new_data.append(new_item)

with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost-train_py.json", "w") as f:
    json.dump(new_data, f, indent = 4)