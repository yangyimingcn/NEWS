# 深度学习MLP新闻文本分类
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
 
 
def load_data(filepath):
    train_df = pd.read_csv(filepath)
    x_train, y_train = train_df['text'], train_df['label']
    x_train, x_test, y_train, y_test = \
        train_test_split(x_train, y_train, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    return x_train, x_val, x_test, y_train, y_val, y_test
 
 
def data_prep(x_train, y_train, x_val, x_test):
    tf_idf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x_train = [" ".join(jieba.lcut(i)) for i in x_train]
    x_test = [" ".join(jieba.lcut(i)) for i in x_test]
    x_train = tf_idf.fit_transform(x_train)
    x_val = tf_idf.transform(x_val)
    x_test = tf_idf.transform(x_test)
    selector = SelectKBest(f_classif, k=min(20000, x_train.shape[1]))
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)  # numpy.float64
    x_val = selector.transform(x_val)
    x_test = selector.transform(x_test)
    return x_train, x_val, x_test
 
 
def main():
    x_train, x_val, x_test, y_train, y_val, y_test = load_data("newses.csv")
    x_train, x_val, x_test = data_prep(x_train, y_train, x_val, x_test)
    model = models.Sequential([
        Dropout(rate=0.2, input_shape=x_train.shape[1:]),  # x_train.shape[1:]：(20000,)
        Dense(units=64, activation='relu'),
        Dropout(rate=0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train.toarray(), y_train, epochs=100, verbose=0,
                        validation_data=(x_val.toarray(), y_val),
                        batch_size=128)
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_accuracy'][-1],
                                                            loss=history['val_loss'][-1]))
    model.evaluate(x_test.toarray(), y_test)
    y_predict = model.predict(x_test.toarray())
    predicts = []
    for i in y_predict:
        predicts.append(np.argmax(i))
    print(f'Predicts：{predicts}')
    score = f1_score(y_test, predicts, average='macro')
    print(f'F1_core：{score}')
    model.save('News_mlp_model.h5')
 
 
main()