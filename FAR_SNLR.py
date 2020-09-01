import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
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
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
def Get_Posc(data,alldata, c,neighborhood_dict):
    POSc = []
    numberSample, numberAttribute = alldata.shape  #numberSample, numberAttribute分别表示数据集的行数和列数                       #所以总列数加一
    num,attr=data.shape
    for i in range(num):
        current_data =data[i, -1]   #当前被比较的行的序号数
        current_label = data[i, -2]    #当前被比较的行的标签（标签在数据集的倒数第二列）
        Mul_Array = np.delete(alldata, current_data, axis=0)  #删除需要做比较的行
        dis = np.sum((Mul_Array[:, 0:numberAttribute - 2] - data[i, 0:numberAttribute - 2]) ** 2, axis=1)
        Neighborhood_object = [current_data]  #存放cre_attribute的邻域的数据的序号
        object = [current_label]   #存放index_attribute领域的数据的标签
        dis_list=list(np.where(dis <=c**2)[0])
        object.extend(list(Mul_Array[dis_list, -2]))  #将符合条件的数据的标签存入列表（标签在数据集的倒数第二列）
        Neighborhood_object.extend([int(i) for i in Mul_Array[dis_list, -1]])
        if len(set(object)) == 1:  #同一邻域域中所有数据标签一致
            POSc.append(Neighborhood_object[0])
            if Neighborhood_object[0] in neighborhood_dict:
                del neighborhood_dict[Neighborhood_object[0]]
        else:
            neighborhood_dict[Neighborhood_object[0]]=Neighborhood_object[1:]
    return sorted(set(POSc)),neighborhood_dict.copy()

def Get_Posc1(data,alldata, c):
    POSc = []
    numberSample, numberAttribute = alldata.shape  #numberSample, numberAttribute分别表示数据集的行数和列数                       #所以总列数加一
    num,attr=data.shape
    for i in range(num):
        current_data =data[i, -1]   #当前被比较的行的序号数
        current_label = data[i, -2]    #当前被比较的行的标签（标签在数据集的倒数第二列）
        Mul_Array = np.delete(alldata, current_data, axis=0)  #删除需要做比较的行
        dis = np.sum((Mul_Array[:, 0:numberAttribute - 2] - data[i, 0:numberAttribute - 2]) ** 2, axis=1)
        Neighborhood_object = [current_data]  #存放cre_attribute的邻域的数据的序号
        object = [current_label]   #存放index_attribute领域的数据的标签
        object.extend(list(Mul_Array[list(np.where(dis <=c**2)[0]), -2]))  #将符合条件的数据的标签存入列表（标签在数据集的倒数第二列）
        if len(set(object)) == 1:  #同一邻域域中所有数据标签一致
            POSc.append(Neighborhood_object[0])
    return sorted(set(POSc))

def res(data,alldata,c):
    s=[]
    alld = Get_Posc1(data, alldata, c)
    numsample,numAttribute=data.shape
    Att=[i for i in range(numAttribute)]
    for i in range(numAttribute-2):
        re=Att.copy()
        re.remove(i)
        a=Get_Posc1(data[:,re], alldata[:,re], c)
        if sorted(alld)!=sorted(a):
            s.append(i)
    return s

def neighborhood_Posc(neighborhood_dict,data,attribute,d):
    POSc=[]
    neighborhood_dict_copy=neighborhood_dict.copy()
    border_flag=True
    neighborhood_dict_num=len(neighborhood_dict_copy)
    i=0
    while i <len(list(neighborhood_dict_copy.keys())) :
        k = list(neighborhood_dict_copy.keys())
        # print(k[i])
        Neighborhood_object = [data[int(k[i]),-1]]  # 存放cre_attribute的邻域的数据的序号
        object_lable = [data[int(k[i]),-2]]
        data_col=data[neighborhood_dict_copy[k[i]]]
        dis = np.sum((data_col[:,attribute[:-2]] - data[int(k[i]), attribute[:-2]]) ** 2, axis=1)
        find_dis=list(np.where(dis <= d ** 2)[0])
        object_lable.extend(list(data_col[find_dis, -2]))
        Neighborhood_object.extend(list(data_col[find_dis, -1]))
        if len(set(object_lable))==1:
            # print(attribute[:-2],Neighborhood_object[0])
            POSc.append(int(Neighborhood_object[0]))
            if Neighborhood_object[0] in neighborhood_dict_copy:
                del neighborhood_dict_copy[Neighborhood_object[0]]
                border_flag=False
        else:
            if len(neighborhood_dict_copy[Neighborhood_object[0]])!=len(Neighborhood_object[1:]):
                border_flag=False
            neighborhood_dict_copy[Neighborhood_object[0]] = [int(j) for j in Neighborhood_object[1:]]
            i+=1
            # print([int(j) for j in Neighborhood_object[1:]])

        # print()
    return sorted(set(POSc)),neighborhood_dict_copy,border_flag

def Neighborhood_relationship(data, alldata, d):
    ans=res(data,alldata,d)
    Attribute = [i for i in range(len(data[0, :]) - 2) if i not in ans]
    all_pos = []
    if len(ans)>0:
        neighborhood_dict = {}
        # ans=ans
        posc,neighborhood_dict = Get_Posc(data[:, sorted(ans)+[-2,-1]], alldata[:, sorted(ans)+[-2,-1]], d,neighborhood_dict.copy())
        all_pos.extend(posc)
        data = np.delete(alldata, all_pos, axis=0)
    else:
        max_num = 0
        max_pos = []
        c = -1
        for i in range(len(Attribute)):
            neighborhood_dict = {}
            ans.append(Attribute[i])
            posc,neighborhood_dict_copy= Get_Posc(alldata[:,[i,-2,-1]],alldata[:,[i,-2,-1]], d,neighborhood_dict.copy())
            posc_num = len(posc)
            if posc_num > max_num:
                max_num = posc_num
                max_pos = posc
                c = Attribute[i]
                neighborhood_dict_max=neighborhood_dict_copy
            ans.remove(Attribute[i])

        if max_num > 0:
            ans.append(c)
            Attribute.remove(c)
            all_pos.extend(max_pos)
            data = np.delete(alldata, all_pos, axis=0)
            neighborhood_dict=neighborhood_dict_max
    ans.extend([len(data[0, :]) - 2, len(data[0, :]) - 1])
    while True:
        if len(Attribute) == 0 or len(data) == 0:
            break
        else:
            max_num = 0
            max_pos = []
            border=[]
            c = -1
            # print(sorted(ans))
            for i in range(len(Attribute)):
                ans.append(Attribute[i])
                # print(sorted(ans))
                neighborhood_dict_num=len(neighborhood_dict)
                posc,neighborhood_dict_copy, border_flag = neighborhood_Posc(neighborhood_dict, alldata, sorted(ans), d)

                # print(neighborhood_dict_copy)
                if float==True:
                    border.append(i)
                posc_num = len(posc)
                # print(Attribute[i],posc_num)
                if posc_num > max_num:
                    max_num = posc_num
                    max_pos = posc
                    c = Attribute[i]
                    neighborhood_dict_max=neighborhood_dict_copy
                ans.remove(Attribute[i])
            # print(c,max_num)
            if max_num > 0:
                ans.append(c)
                Attribute.remove(c)
                for attribute in border:
                    Attribute.remove(attribute)
                all_pos.extend(max_pos)
                data = np.delete(alldata, all_pos, axis=0)
                neighborhood_dict = neighborhood_dict_max
                # print(neighborhood_dict_max.keys())
            else:
                break
    return sorted(ans)[:-2]


def main():
    warnings.filterwarnings("ignore")  # 忽略警告
    # keys = ['horse']
    keys = ['anneal', 'credit', 'german','heart1', 'hepatitis', 'horse', 'iono',
                    'wdbc','wine','lymphography', 'zoo']
    for d in range(len(keys)):
        print(keys[d])
        df = pd.read_csv("D:\\code\\UCI\\" + keys[d] + ".csv", header=None)
        data = df.values
        numberSample, numberAttribute = data.shape
        # print(numberSample)
        minMax = MinMaxScaler()  # 将数据进行归一化
        data = np.hstack((minMax.fit_transform(data[:, 1:]), data[:, 0].reshape(numberSample, 1)))
        num, dim = data[:, :-1].shape
        index = np.array([int(i) for i in range(0, num)]).reshape(num, 1)  # 索引列
        data = np.hstack((data, index))  # 加上索引列
        Neighborhood_relationship(data, data, 0.16)
        # if keys[d] == 'anneal':
        #     i = 100
        #     while i < 701:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.16)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 100
        #     start = time.clock()
        #     Neighborhood_relationship(data, data, 0.16)
        #     end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     print(end - start)
        #     # start = time.clock()
        #     # relative_attributes_reduction1(data, 0.25)# iono 0.03 glass 0.02 wine 0.14 \Sonar 0.1 pima 0.01 wdbc 0.13
        #     # # get_attribute_reduction(dataset)
        #     # end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end-start)
        # if keys[d] == 'credit':
        #     i = 69
        #     while i < 691:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.16)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 69
        #     # start = time.clock()
        #     # re = rough_1(data, data, Equivalence_class)
        #     # end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     # print(re)
        # if keys[d] == 'heart1':
        #     i = 27
        #     while i < 271:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.19)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 27
        #     # start = time.clock()
        #     # re = rough_1(data, data, Equivalence_class)
        #     # end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     # print(re)
        # if keys[d] == 'hepatitis':
        #     i = 15
        #     while i < 151:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.25)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 10
        #     start = time.clock()
        #     Neighborhood_relationship(data, data, 0.25)
        #     end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     print(end - start)
        # if keys[d] == 'horse':
        #     i = 36
        #     while i < 361:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.19)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 36
        #     start = time.clock()
        #     Neighborhood_relationship(data, data, 0.19)
        #     end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     print(end - start)
        # if keys[d] == 'iono':
        #     i = 35
        #     while i < 351:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.22)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end-start)
        #         i = i + 35
        #     # start = time.clock()
        #     # re = rough_1(data, data, Equivalence_class)
        #     # end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     # print(re)
        # if keys[d] == 'wdbc':
        #     i = 56
        #     while i < 561:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.19)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 56
        #     start = time.clock()
        #     Neighborhood_relationship(data, data, 0.19)
        #     end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     print(end - start)
        # if keys[d] == 'zoo':
        #     i = 10
        #     while i < 101:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.16)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 10
        # if keys[d] == 'german':
        #     i = 100
        #     while i < 1001:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.16)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 100
        # if keys[d] == 'wine':
        #     i = 17
        #     while i < 171:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.37)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 17
        #     start = time.clock()
        #     Neighborhood_relationship(data, data, 0.37)
        #     end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     print(end - start)
        # if keys[d] == 'lymphography':
        #     i = 14
        #     while i < 140:
        #         dataset = data[0:i, :]
        #         start = time.clock()
        #         Neighborhood_relationship(dataset, dataset, 0.37)
        #         end = time.clock()
        #         # print("对象个数", i, "运行时间", end - start)
        #         print(end - start)
        #         i = i + 14
        #     start = time.clock()
        #     Neighborhood_relationship(data, data, 0.37)
        #     end = time.clock()
        #     # print("对象个数", len(data), "运行时间", end - start)
        #     print(end - start)
        # # for i in range(1,51):
        # #     # p=Get_Posc1(data, data, i*0.01)
        # #     start=time.clock()
        # #     a=Neighborhood_relationship(data, data, i*0.01)
        # #     end=time.clock()
        # #     # a=res(data,data,i*0.01)
        # #     print(i*0.01,a)
        # #     print(end-start)

if __name__ == '__main__':
    main()