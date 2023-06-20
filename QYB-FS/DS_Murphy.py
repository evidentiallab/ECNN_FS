import numpy as np
# x_result = {"A": 0, "A|B": 0, "Ω": 0, "empty": 0}
def change_to_dict(x_pred,num_classes):
    dict_list = []
    for i in range(x_pred.shape[0]):
        row_dict = {}
        for j in range(x_pred.shape[1]):
            if j == num_classes:
                key=""
                for k in range (num_classes):
                    key=key+str(k)
            else:
                key = str(j)
            value = x_pred[i,j]
            row_dict[key] = value
        dict_list.append(row_dict)
    return dict_list

def focal_Element(m_dict):
    key_m = []
    m_key = m_dict.keys()
    for item in m_key:
        key_m.append(item)
    key_m_set = set(key_m)
    return key_m_set


def computrEmpty(m_1, m_2):
    F1 = focal_Element(m_1)
    F2 = focal_Element(m_2)
    # 空集合
    theta_set1 = set()
    theta_set2 = set()
    empty = set()
    m_empty = 0.00
    for item1 in F1:
        theta_set1.update(item1)
        for item2 in F2:
            theta_set2.update(item2)
            if theta_set1.intersection(theta_set2) == empty:
                m_empty += m_1[item1] * m_2[item2]
            theta_set2.clear()
        theta_set1.clear()
    return m_empty


def computeInter(m_1, m_2):
    F1 = focal_Element(m_1)
    F2 = focal_Element(m_2)
    thetaSet = focal_Element({"A": 0, "B": 0, "C": 0, "D": 0})
    # 空集合
    f1_set = set()
    f2_set = set()
    strSet = set()
    inter = dict()
    for item0 in thetaSet:
        strSet.update(item0)
        sum_ = 0.00
        for item1 in F1:
            f1_set.update(item1)
            for item2 in F2:
                f2_set.update(item2)
                if f1_set.intersection(f2_set) == strSet:
                    sum_ += m_1[item1] * m_2[item2]
                f2_set.clear()
            f1_set.clear()
        inter[item0] = sum_
        strSet.clear()
    return inter


def weightAvg(n, avgSet):
    result = {}  # 定义一个空字典，用于保存计算结果
    for i in range(1, n):
        if i == 1:
            diedai = computeInter(avgSet, avgSet)
            k = computrEmpty(avgSet, avgSet)
            k = 1/(1-k)
            for item in diedai.keys():
                value_ = round(diedai[item]*k, 4)
                diedai[item] = value_
            diedai_0 = diedai
            # print("第{}次迭代结果：{}".format(i, diedai_0))
        else:
            diedai_1 = computeInter(diedai_0, avgSet)
            k = computrEmpty(diedai_0, avgSet)
            k = 1/(1-k)
            for item in diedai_1.keys():
                value_ = round(diedai_1[item]*k, 4)
                diedai_1[item] = value_
            diedai_0 = diedai_1
            # print("第{}次迭代结果：{}".format(i, diedai_1))
    result['diedai'] = diedai_0   # 将最终的命题置信度保存在字典中
    result['k'] = k              # 将空集合的量 k 保存在字典中
    return result                # 返回字典结果


# if __name__ == '__main__':
#     n = 5
#     #average_ = {"A": 0.4872, "AC": 0.2260, "C": 0.0790, "B": 0.2078}
#     #result=weightAvg(n, average_)
#     # avgSet1={"0": 0.4, "01": 0.2, "2": 0.4}
#     # avgSet2={"0": 0.7, "012": 0.3}
#     # k=computrEmpty(avgSet1, avgSet2)
#     # print(k)
#     arr1=[4.28373646e-03 8.77304527e-04 7.58930109e-04 2.36356034e-04
#             1.41467332e-04 9.75778282e-01 1.18084921e-04 9.44449380e-03
#             1.73666840e-03 2.67764926e-03 3.94712016e-03]

#     arr2=[[0.1,0.8,0.1],[0.6,0.3,0.1]]
#     arr1=np.array(arr1)
#     arr2=np.array(arr2)
#     arr_to_dict1=change_to_dict(arr1,2)
#     arr_to_dict2=change_to_dict(arr2,2)
#     k=[]
#     for n in range(arr1.shape[0]):
#         k.append(computrEmpty(arr_to_dict1[n],arr_to_dict2[n]))
#     print(k)













