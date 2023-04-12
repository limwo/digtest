import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 读入文本文件
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 定义停用词，可以根据实际情况增加或修改
stopwords = set(['的', '了', '是', '和', '就', '在', '也', '有', '我', '你'])

# 将文本分词并去除停用词
words = [word for word in text.split() if word not in stopwords]

# 统计每个单词出现的次数
word_counts = {}
for word in words:
    if word not in word_counts:
        word_counts[word] = 0
    word_counts[word] += 1

# 生成词云图
mask = np.array(Image.open("mask.png"))  # 读入词云图形状
wc = WordCloud(background_color="white", max_words=1000, mask=mask, contour_width=3, contour_color='steelblue')
wc.generate_from_frequencies(word_counts)

# 显示词云图
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# 保存词云图
wc.to_file("wordcloud.png")
