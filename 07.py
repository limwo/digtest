import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# 加载BERT预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = TFBertModel.from_pretrained('bert-base-chinese')

# 定义Siamese模型
input1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input1')
input2 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input2')
bert_output1 = bert_model(input1)[1]
bert_output2 = bert_model(input2)[1]
cosine_similarity = tf.keras.layers.Dot(axes=[1, 1], normalize=True)([bert_output1, bert_output2])
siamese_model = tf.keras.models.Model(inputs=[input1, input2], outputs=cosine_similarity)

# 编译模型
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 准备训练数据和标签
text1 = '我喜欢看电影'
text2 = '看电视剧很有趣'
text3 = '跑步是健身的一种方式'
text4 = '健身很有益于身体健康'
train_data = [    (text1, text2, 0),    (text1, text3, 0),    (text2, text4, 0),    (text1, text4, 1),    (text2, text3, 0),    (text3, text4, 1)]
train_texts1 = [data[0] for data in train_data]
train_texts2 = [data[1] for data in train_data]
train_labels = [data[2] for data in train_data]

# 将文本转换为BERT模型输入格式
train_encodings1 = tokenizer(train_texts1, truncation=True, padding=True)
train_encodings2 = tokenizer(train_texts2, truncation=True, padding=True)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings1), dict(train_encodings2), train_labels)).shuffle(100).batch(2)

# 训练模型
siamese_model.fit(train_dataset, epochs=3)

# 准备测试数据
test_data = [    (text1, text2),    (text1, text3),    (text2, text4),    (text1, text4),    (text2, text3),    (text3, text4)]
test_texts1 = [data[0] for data in test_data]
test_texts2 = [data[1] for data in test_data]

# 将文本转换为BERT模型输入格式
test_encodings1 = tokenizer(test_texts1, truncation=True, padding=True)
test_encodings2 = tokenizer(test_texts2, truncation=True, padding=True)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings1), dict(test_encodings2))).batch(2)

# 预测相似度
predictions = siamese_model.predict(test_dataset)

# 输出相似度
for i in range(len(test_data)):
    print(test_data[i], '相似度:', predictions[i][0])
