from wordcloud import WordCloud
import numpy as np

# 读取数据
reportpath = './思政元素.txt'
report19 = open(reportpath, 'r', encoding='UTF-8').read()

# 以空格为分隔符进行分词
words = report19.split()

# 将分词结果保存在字典中进行词频统计
word_freq = {}
for word in words:
    if word in word_freq:
        word_freq[word] += 1
    else:
        word_freq[word] = 1

# 将字典按词频排序
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# 取前30个高频词
top30_words = sorted_word_freq[:30]

# 将列表转换为numpy数组并作矩阵转置，方便画图取用
usedata = np.array(top30_words).T

# 生成词云图
wc = WordCloud(font_path='msyh.ttc', background_color='white', width=800, height=600, max_words=30)
wc.generate_from_frequencies(dict(top30_words))
wc.to_file("wordcloud.png")

# 输出词频统计结果
print(usedata)
