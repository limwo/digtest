from scipy.spatial.distance import cityblock

...

# 3. 计算目标句子与文档中所有句子的相似度
similarity_scores = []
for sentence in sentences:
    # 对每个句子进行BERT编码
    sentence_encoding = tokenizer.encode_plus(sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    # 计算目标句子与当前句子的相似度得分
    similarity_score = 1 / (1 + cityblock(model(target_encoding['input_ids'])[1].detach().numpy(),
                                          model(sentence_encoding['input_ids'])[1].detach().numpy()).mean())
    similarity_scores.append(similarity_score)

...
