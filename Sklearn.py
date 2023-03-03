# 机器学习sklearn新闻文本分类
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import time
 
 
def load_data(filepath):
    train_df = pd.read_csv(filepath)
    x_train, y_train = train_df['text'], train_df['label']
    x_train, x_test, y_train, y_test = \
        train_test_split(x_train, y_train, test_size=0.2)
    return x_train, x_test, y_train, y_test
 
 
def data_prep(x_train, y_train, x_test):
    tf_idf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x_train = [" ".join(jieba.lcut(i)) for i in x_train]
    x_test = [" ".join(jieba.lcut(i)) for i in x_test]
    x_train = tf_idf.fit_transform(x_train)
    x_test = tf_idf.transform(x_test)
    selector = SelectKBest(f_classif, k=min(20000, x_train.shape[1]))
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)  # numpy.float64
    x_test = selector.transform(x_test)
    return x_train, x_test
 
 
def main():
    x_train, x_test, y_train, y_test = load_data("newses.csv")
    x_train, x_test = data_prep(x_train, y_train, x_test)
 
    start = time.time()
    gnb = GaussianNB()  # 朴素贝叶斯
    print(f'1：{cross_val_score(gnb, x_train.toarray(), y_train, cv=10)}')
    gnb.fit(x_train.toarray(), y_train)
    answer_gnb = pd.Series(gnb.predict(x_test.toarray()))
    answer_gnb.to_csv("answer_gnb.csv", header=False, index=False)
    score_gnb = f1_score(y_test, answer_gnb, average='macro')
    print(f'F1_core_gnb：{score_gnb}')
    end = time.time()
    print(f'时间：{end - start}s')
 
    start = time.time()
    rc = RidgeClassifier()  # 岭回归分类器
    print(f'\n2：{cross_val_score(rc, x_train, y_train, cv=10)}')
    rc.fit(x_train, y_train)
    answer_rc = pd.Series(rc.predict(x_test))
    answer_rc.to_csv("answer_rc.csv", header=False, index=False)
    score_rc = f1_score(y_test, answer_rc, average='macro')
    print(f'F1_core_rc：{score_rc}')
    end = time.time()
    print(f'时间：{end - start}s')
 
    start = time.time()
    sv = svm.SVC()  # 支持向量机
    print(f'\n3：{cross_val_score(sv, x_train, y_train, cv=10)}')
    sv.fit(x_train, y_train)
    answer_sv = pd.Series(sv.predict(x_test))
    answer_sv.to_csv("answer_sv.csv", header=False, index=False)
    score_sv = f1_score(y_test, answer_sv, average='macro')
    print(f'F1_core_sv：{score_sv}')
    end = time.time()
    print(f'时间：{end - start}s')
 
 
main()
