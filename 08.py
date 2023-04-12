import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载预训练好的Bert模型和tokenizer
model = AutoModel.from_pretrained('bert-base-chinese')
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

# 2. 准备一个目标句子和一个待比较文档，将文档分成句子列表
target_sentence = '99.水资源环境管理的途径和方法水资源环境管理原则；完善管理体制和管理组织机构，加强水资源的统一管理；树立水环境污染有偿使用的水权观念，并将其引入水资源管理；实行水污染物总量控制，推行许可证制度，实现水量和水质并重管建设，积极开发新水源'
with open("test0.txt", "r", encoding="utf-8") as f_in:
    # 将文档中每遇到一个句号就换行
    document = "\n".join([sent.strip() for sent in f_in.read().split("\n") if len(sent) > 0])

# 将处理后的文档保存到中间文件中
with open("document.txt", "w", encoding="utf-8") as f_out:
    f_out.write(document)

# 使用readlines()函数读入处理后的文档
with open("document.txt", "r", encoding="utf-8") as f:
    sentences = f.readlines()

# 对目标句子进行BERT编码
target_encoding = tokenizer.encode_plus(target_sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

# 3. 计算目标句子与文档中所有句子的相似度
similarity_scores = []
for sentence in sentences:
    # 对每个句子进行BERT编码
    sentence_encoding = tokenizer.encode_plus(sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    # 计算目标句子与当前句子的相似度得分
    similarity_score = -torch.dist(model(target_encoding['input_ids'])[1], model(sentence_encoding['input_ids'])[1], p=2).item()
    similarity_scores.append(similarity_score)

# 4. 输出相似度前三的句子及其相似度得分
sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:3]


for i, sentence in enumerate(sentences):
    print(f"第{i+1}个句子的相似度为{similarity_scores[i]}")

print("相似度前三的句子为：")
for i in sorted_indices:
 print(f"第{i+1}个句子是'{sentences[i].strip()}'，与目标句子的相似度为{similarity_scores[i]}")
