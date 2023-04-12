import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载预训练好的Bert模型和tokenizer
model = AutoModel.from_pretrained('bert-base-chinese')
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

# 2. 准备一个目标句子和一个待比较文档，将文档分成句子列表
target_sentence = '这是目标句子。'
with open("test.txt", "r", encoding="utf-8") as f_in:
    # 将文档中每遇到一个句号就换行
    document = "\n".join([sent.strip() for sent in f_in.read().split("。") if len(sent) > 0])

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
    similarity_score = cosine_similarity(model(target_encoding['input_ids'])[1].detach().numpy(), 
                                          model(sentence_encoding['input_ids'])[1].detach().numpy()).mean()
    similarity_scores.append(similarity_score)

# 4. 输出所有句子的相似度得分
for i, sentence in enumerate(sentences):
    print(f"第{i+1}个句子的相似度为{similarity_scores[i]}")

# 找到相似度最高的句子及其相似度得分
max_index = similarity_scores.index(max(similarity_scores))
most_similar_sentence = sentences[max_index]
most_similar_score = similarity_scores[max_index]

print(f"\n文档中最相似的句子是'{most_similar_sentence}'，与目标句子的相似度为{most_similar_score}")
