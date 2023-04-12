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

# 按照单词出现次数从大到小排序
sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# 输出前10个出现次数最多的单词
for i in range(10):
    print(f"{sorted_word_count[i][0]}: {sorted_word_count[i][1]}")
