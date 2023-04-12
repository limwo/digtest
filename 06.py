import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 两个文本
text1 = "这是一个文本示例"
text2 = "这是另外一个示例文本"

# 构建词向量矩阵
vectorizer = CountVectorizer().fit_transform([text1, text2])
vectors = vectorizer.toarray()

# 计算余弦相似度
similarity = cosine_similarity(vectors[0].reshape(1,-1), vectors[1].reshape(1,-1))

print(similarity)
