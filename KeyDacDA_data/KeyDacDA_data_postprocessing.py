import json

# Read the JSON file
with open("/home/ysnamgoong42/ws/XLCoST/KeyDacDA_data/xlcost-train_py-keydac-only_positive.json", 'r') as f:
    xlcost_train = json.load(f)

# Replace all '\n' tokens in the value of the key "code"
for d in xlcost_train:
    d['code'] = d['code'].replace('\n', 'NEW_LINE')

# Replace all '\n' tokens in the value of the key "doc"
for d in xlcost_train:
    d['doc'] = d['doc'].replace('\n', '')

print("-") # check xlcost_train

# Write the "code" values to a Python file
with open("train-Python-desc-tok_KeyDacDA.py", "w") as f:
    for item in xlcost_train:
        f.write(item['code'] + '\n')

# Write the "doc" values to a text file
with open("train-Python-desc-tok_KeyDacDA.txt", "w") as f:
    for item in xlcost_train:
        f.write(item['doc'] + '\n')

