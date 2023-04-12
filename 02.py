import wordcloud
from PIL import Image
import numpy as np

# 打开文件并读取文本内容
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 将文本内容按空格分割成单词列表
words = text.split()

# 定义一个字典，用于存储每个单词的出现次数
word_count = {}

# 遍历单词列表，统计每个单词出现的次数
for word in words:
    if word not in word_count:
        word_count[word] = 1
    else:
        word_count[word] += 1

# 生成词云图
wc = wordcloud.WordCloud(background_color="white", max_words=100, width=800, height=400, font_path="msyh.ttc")
wc.generate_from_frequencies(word_count)

# 打开图像文件并读取图像数据
mask = np.array(Image.open("mask.png"))

# 根据图像数据生成形状词云图
wc_masked = wordcloud.WordCloud(background_color="white", max_words=100, width=800, height=400, font_path="msyh.ttc", mask=mask)
wc_masked.generate_from_frequencies(word_count)

# 保存词云图到文件
wc.to_file("wordcloud.png")
wc_masked.to_file("wordcloud_masked.png")
