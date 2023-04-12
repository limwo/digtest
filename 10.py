import torch
from transformers import AutoTokenizer, AutoModel

# 加载预训练的BERT模型和tokenizer
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 准备两个句子作为示例
text1 = "I love playing basketball"
text2 = "Basketball is my favorite sport"

# 对两个句子进行BERT编码
text1_encoding = tokenizer.encode_plus(text1, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
text2_encoding = tokenizer.encode_plus(text2, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

# 计算两个句子之间的相似度得分
distance = torch.dist(model(text1_encoding['input_ids'])[1], model(text2_encoding['input_ids'])[1], p=2).item()
similarity_score = 1 / (1 + distance)

print(f"文本1：{text1}")
print(f"文本2：{text2}")
print(f"相似度得分：{similarity_score}")

