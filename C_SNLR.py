import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings("ignore")  # 忽略警告
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
def Get_Posc(data,equivalence_class):
    """
    :param data: 传入的数据集
    :param equivalence_class:决策属性的等价类
    :return: 返回得出的正域结果
    """
    dataset_class=equivalence_class
    Equivalence_class=[]
    Equivalence_class_set=[]
    Posc=[]
    for i in np.unique(data[:,:-2], axis=0):
        for j in range(len(data)):
            if (i==data[j,:-2]).all():
                Equivalence_class.append(int(data[j,-1]))
        Equivalence_class_set.append(Equivalence_class)
        Equivalence_class=[]
    for j in dataset_class:
        for k in Equivalence_class_set:
            if set(k)<=set(j):
                Posc.extend(k)
    return Posc

def rough_set(data,Equivalence_class):
    """
    第一步计算冗余属性
    :param data:
    :param Equivalence_class:
    :return: 冗余属性
    """
    re_attribute=[]
    samples_num, Attribute_num = data[:, :-2].shape
    Attribute_list=[i for i in range(Attribute_num)]
    new_data = data
    re_list={}
    all_bor=[]
    re_bor=[]
    if len(new_data) != 0:
        init_list = [new_data]
        for i in Attribute_list:
            # print(i)
            re_list[i] = init_list
            re_init_list = []
            for dataset in init_list:
                dict = {}
                dataset = dataset[dataset[:, i].argsort()]
                for j in set(dataset[:, i]):
                    dict[j] = []
                for sample in dataset:
                    dict[sample[i]].append(sample)
                for values in dict.values():
                    if len(set(np.array(values)[:, -2])) != 1:
                        re_init_list.append(np.array(values))
            init_list = re_init_list
        for i in init_list:
            all_bor.extend(i[:,-1])
    for k in Attribute_list:
        re_bor=[]
        init_list =re_list[k]
        for i in Attribute_list[k+1:]:
            re_init_list = []
            for dataset in init_list:
                dict = {}
                dataset = dataset[dataset[:, i].argsort()]
                for j in set(dataset[:, i]):
                    dict[j] = []
                for sample in dataset:
                    dict[sample[i]].append(sample)
                for values in dict.values():
                    if len(set(np.array(values)[:, -2])) != 1:
                        re_init_list.append(np.array(values))

            init_list = re_init_list
        for i in init_list:
            re_bor.extend(i[:,-1])
        if sorted(re_bor)!=sorted(all_bor):
            re_attribute.append(k)
    return re_attribute

def rough_set_attribute(data,Equivalence_class):
    re = rough_set(data, Equivalence_class)
    new_data=data
    if len(new_data) != 0:
        C = [i for i in range(len(data[0, :-2]))]
        CC = sorted(list(set(C).difference(set(re))))
        init_list = [new_data]
        for i in re:
            re_init_list = []
            for dataset in init_list:
                dict = {}
                dataset = dataset[dataset[:, i].argsort()]
                for j in set(dataset[:, i]):
                    dict[j] = []
                for sample in dataset:
                    dict[sample[i]].append(sample)
                for values in dict.values():
                    if len(set(np.array(values)[:, -2])) != 1:
                        re_init_list.append(np.array(values))
            init_list = re_init_list
        while CC:
            border_list=[]
            min_num=len(data)
            max_init_list=[]
            not_redu = 0
            for i, q in enumerate(CC):
                re_init_list = []
                border_num=0
                for dataset in init_list:
                    dict = {}
                    dataset = dataset[dataset[:, q].argsort()]
                    for j in set(dataset[:, q]):
                        dict[j] = []
                    for sample in dataset:
                        dict[sample[q]].append(sample)
                    for values in dict.values():
                        if len(set(np.array(values)[:, -2])) != 1:
                            re_init_list.append(np.array(values))
                            border_num+=len(np.array(values))
                border_list.append(border_num)
                if min_num>border_num:
                    min_num=border_num
                    max_init_list=re_init_list
                    not_redu=q
            if len(set(border_list))==1:
                break
            CC.remove(not_redu)
            re.append(not_redu)
            init_list=max_init_list
            if len(init_list) == 0:
                break
    return sorted(re)


def mean_std(a):
    # 计算一维数组的均值和标准差
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std



def fit(path):
    warnings.filterwarnings("ignore")  # 忽略警告
    df = pd.read_csv(path, header=None)
    data = df.values
    k = 2
    numberSample, numberAttribute = data.shape
    d1 = data[:, 0].reshape(numberSample, 1)
    for i in range(1, len(data[0])):
        if len(set(data[:, i])) > 10:
            d2 = np.array([pd.cut(data[:, i], k, labels=range(k))])
            d1 = np.hstack((d1, d2.reshape(numberSample, 1)))
        else:
            d1 = np.hstack((d1, data[:, i].reshape(numberSample, 1)))
    data = np.hstack((d1[:, 1:], d1[:, 0].reshape(numberSample, 1)))
    orderAttribute = np.array([i for i in range(0, numberSample)]).reshape(numberSample,
                                                                           1)  # 创建一个列表保存数字序列从1到numberSample
    data = np.hstack((data, orderAttribute))  # 将列表orderAttribute加在原数据集后面
    Attribute = data[:, -2]
    samples = data[:, -1]
    numberSample, numberAttribute = data.shape

    Equivalence_class = []
    samples = data[:, -1]
    for q in set(Attribute):
        Equivalence_class.append(samples[Attribute == q])
    return data,Equivalence_class

data,Equivalence_class=fit(path)
results=rough_set_attribute(data,Equivalence_class)