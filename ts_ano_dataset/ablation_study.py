import json


# wo COT
# OUTPUT_LABEL = "/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3-woCOTDATA.json"
# OUTPUT_DATASET = "/home/wzx/LLaMA-Factory/data/sft-zscore-new3-woCOTDATA.json"

# with open('/home/wzx/LLaMA-Factory/data/sft-zscore-new3.json', 'r') as f:
#     result = json.load(f)

# with open('/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3.json', 'r') as f:
#     label = json.load(f)

# result_new = []
# label_new = []
# for r, l in zip(result, label):
#     fields = l['fields']
#     key = list(fields.keys())[0]

#     if fields[key] == 'COT':
#         continue
#     else:
#         result_new.append(r)
#         label_new.append(l)
# json.dump(label_new, open(OUTPUT_LABEL, 'wt'), ensure_ascii=False, indent=4)
# json.dump(result_new, open(OUTPUT_DATASET, 'wt'), ensure_ascii=False, indent=4)

# wo indexlength
# OUTPUT_LABEL = "/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3-woINDEXLENGTH.json"
# OUTPUT_DATASET = "/home/wzx/LLaMA-Factory/data/sft-zscore-new3-woINDEXLENGTH.json"

# with open('/home/wzx/LLaMA-Factory/data/sft-zscore-new3.json', 'r') as f:
#     result = json.load(f)

# with open('/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3.json', 'r') as f:
#     label = json.load(f)

# result_new = []
# label_new = []
# for r, l in zip(result, label):
#     fields = l['fields']
#     key = list(fields.keys())[0]
#     if key == 'index' or key == 'length':
#         continue
#     else:
#         result_new.append(r)
#         label_new.append(l)
# json.dump(label_new, open(OUTPUT_LABEL, 'wt'), ensure_ascii=False, indent=4)
# json.dump(result_new, open(OUTPUT_DATASET, 'wt'), ensure_ascii=False, indent=4)

# wo minmax
# OUTPUT_LABEL = "/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3-wominmax.json"
# OUTPUT_DATASET = "/home/wzx/LLaMA-Factory/data/sft-zscore-new3-wominmax.json"

# with open('/home/wzx/LLaMA-Factory/data/sft-zscore-new3.json', 'r') as f:
#     result = json.load(f)

# with open('/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3.json', 'r') as f:
#     label = json.load(f)

# result_new = []
# label_new = []
# for r, l in zip(result, label):
#     fields = l['fields']
#     key = list(fields.keys())[0]
#     if key == 'min' or key == 'max':
#         continue
#     else:
#         result_new.append(r)
#         label_new.append(l)
# json.dump(label_new, open(OUTPUT_LABEL, 'wt'), ensure_ascii=False, indent=4)
# json.dump(result_new, open(OUTPUT_DATASET, 'wt'), ensure_ascii=False, indent=4)


# wo trend period noise
# OUTPUT_LABEL = "/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3-wotrendperiodnoise.json"
# OUTPUT_DATASET = "/home/wzx/LLaMA-Factory/data/sft-zscore-new3-wotrendperiodnoise.json"

# with open('/home/wzx/LLaMA-Factory/data/sft-zscore-new3.json', 'r') as f:
#     result = json.load(f)

# with open('/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3.json', 'r') as f:
#     label = json.load(f)

# result_new = []
# label_new = []
# for r, l in zip(result, label):
#     fields = l['fields']
#     key = list(fields.keys())[0]
#     if key == 'period' or key == 'noise' or key == 'trend':
#         continue
#     else:
#         result_new.append(r)
#         label_new.append(l)
# json.dump(label_new, open(OUTPUT_LABEL, 'wt'), ensure_ascii=False, indent=4)
# json.dump(result_new, open(OUTPUT_DATASET, 'wt'), ensure_ascii=False, indent=4)


OUTPUT_LABEL = "/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3-wolocal.json"
OUTPUT_DATASET = "/home/wzx/LLaMA-Factory/data/sft-zscore-new3-wolocal.json"

with open('/home/wzx/LLaMA-Factory/data/sft-zscore-new3.json', 'r') as f:
    result = json.load(f)

with open('/home/wzx/LLaMA-Factory/ts_ano_dataset/labels/sft-zscore-new3.json', 'r') as f:
    label = json.load(f)

result_new = []
label_new = []
for r, l in zip(result, label):
    fields = l['fields']
    key = list(fields.keys())[0]
    if key == 'local':
        continue
    else:
        result_new.append(r)
        label_new.append(l)
json.dump(label_new, open(OUTPUT_LABEL, 'wt'), ensure_ascii=False, indent=4)
json.dump(result_new, open(OUTPUT_DATASET, 'wt'), ensure_ascii=False, indent=4)
