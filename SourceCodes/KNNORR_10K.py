import xlwt
import xlrd
import numpy as np
import pandas as pd
from collections import OrderedDict
from copy import deepcopy
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from pathlib import Path
from time import time
from itertools import combinations



class FARI():
    def __init__(self):
        self.K = 5
        self.P = 5

    def D(self,a,b):
        return np.linalg.norm(a-b,ord=2)

    def fit(self,Dist_Matrix, y, abslabeled, Relat_info):
        self.DM = Dist_Matrix
        self.y = y
        self.labels = np.sort(np.unique(self.y))
        self.abslabeled = abslabeled    #对应全局标签
        self.nLabeled = len(abslabeled)
        self.RI = Relat_info    ##对应全局标签
        self.nClass = len(self.labels)
        self.K = self.nClass
        if len(Relat_info) < self.nClass:
            self.P = len(Relat_info)
        else:
            self.P = self.nClass
        return self

    def predict(self, test_ids):
        y_pred = []
        nTest = len(test_ids)
        for idx in test_ids:
            obj_function = OrderedDict()
            for lab in self.labels:
                obj_function[lab] = 0.0

            # # #  # # #  # # #  # # #  # # #  # # #  # # #  # # #
            # --------测试样本多已标记样本的距离---------
            dist_list = np.zeros(self.nLabeled)
            for j, jdx in enumerate(self.abslabeled):
                dist_list[j] = self.DM[idx,jdx]
            # ----------对距离列表排序，目的为寻找近邻---------------
            ord_ids = np.argsort(dist_list)
            # ------为了计算权重，需要计算最大近邻距离和最小近邻距离---------
            max_d = dist_list[ord_ids[self.K-1]]  # 到第K个近邻的距离
            min_d = dist_list[ord_ids[0]]         # 到第1个近邻的距离

            # ------初始化权重列表-------
            KNN_weight = np.zeros(self.K)
            # ----初始化近邻索引列表------
            KNN_index = np.ones(self.K,dtype=int)
            # ------分配近邻索引，计算每个近邻权重--------
            for r, rdx in enumerate(ord_ids[:self.K]):
                KNN_index[r] = int(self.abslabeled[rdx])
                KNN_weight[r] = (max_d - dist_list[ord_ids[r]])/(max_d - min_d)

            # # #  # # #  # # #  # # #  # # #  # # #  # # #
            # ------初始化样本对列表---------
            RI_list = []
            # ----将样本对存入列表------
            for key in self.RI.keys():
                RI_list.append(key)

            # ------------每个测试样本和它的近邻组成样本对----------------
            for r, rdx in enumerate(KNN_index):
                target = self.y[rdx]
                # -------（idx，rdx）到每个样本对的距离---------
                cd_list = []
                for ele in RI_list:
                    cd_list.append(self.DM[idx, ele[0]] + self.DM[rdx,ele[1]])

                # -------对（idx，rdx）到每个样本对的距离进行排序---------
                cd_ord_ids = np.argsort(cd_list)

                # -----------计算（idx，rdx）到其近邻的最大距离和最小距离，用于权重计算----------------
                max_cd = cd_list[cd_ord_ids[self.P-1]]
                min_cd = cd_list[cd_ord_ids[0]]

                # --------------初始化权重列表-------------------
                PNN_weight = np.zeros(self.P)
                PNN_object = []
                for p in range(self.P):
                    PNN_object.append(RI_list[cd_ord_ids[p]])
                    PNN_weight[p] = (max_cd - cd_list[cd_ord_ids[p]])/(max_cd - min_cd)
                # print("PNN_weight::",PNN_weight)
                # print("PNN_object::",PNN_object)

                for q, object in enumerate(PNN_object):

                    if self.RI[object] == ">":
                        Ljq = target
                        Rjq = self.labels[-1]
                    elif self.RI[object] == "<":
                        Ljq = self.labels[0]
                        Rjq = target

                    # print("左边：：",Ljq)
                    # print("右边：：",Rjq)
                    for lab in self.labels:
                        obj_function[lab] += KNN_weight[r] * PNN_weight[q] * (abs(Ljq - lab) + abs(Rjq-lab))
                        # print("目标:",obj_function[lab])

            # print("目标函数：",obj_function)
            y_pred.append(min(obj_function,key=obj_function.get))

        # print(y_pred)
        return y_pred




if __name__ == '__main__':
    p = Path("E:\EEEEE\Dataset")

    names_list = ["SWD","Car","Automobile","Cleveland","Housing-5bin","Stock-5bin","Computer-5bin","Winequality-red","Obesity","Housing-10bin","Stock-10bin","Computer-10bin"]

    class results():
        def __init__(self):
            self.MZEList = []
            self.MAEList = []
            self.F1List = []

    method = "KNNARI"
    for name in names_list:
        print("########################{}".format(name))

        data_path = Path("E:\EEEEE\Dataset")
        partition_path = Path(r"E:\EEEEE\Partitions")

        """--------------read the whole data--------------------"""
        read_data_path = data_path.joinpath(name + ".csv")
        data = np.array(pd.read_csv(read_data_path, header=None))
        X = np.asarray(data[:, :-1], np.float64)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data[:, -1]
        y -= y.min()
        Dist_Matrix = pairwise_distances(X=X,metric='euclidean')
        nClass = len(np.unique(y))


        """--------read the partitions--------"""
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)

        # --------------------------------------
        RESULT = results()
        # --------------------------------------

        workbook = xlwt.Workbook()
        count = 0
        MZE_list = []
        MAE_list = []
        F1_list = []
        for SN in book_partition.sheet_names():
            Budget = 10 * nClass
            S_Time = time()
            train_idx = []
            test_idx = []
            labeled = []
            table_partition = book_partition.sheet_by_name(SN)
            for idx in table_partition.col_values(0):
                if isinstance(idx,float):
                    train_idx.append(int(idx))
            for idx in table_partition.col_values(1):
                if isinstance(idx,float):
                    test_idx.append(int(idx))
            for idx in table_partition.col_values(2):
                if isinstance(idx,float):
                    labeled.append(int(idx))
            labeled = np.array(train_idx)[labeled]

            # ---------Get the Relative information for KNNARI model-------------
            # ----initialize the poo_set-----
            pool_set = deepcopy(train_idx)
            for idx in labeled:
                pool_set.remove(idx)
            # ---select the query pairs-----
            Relat_Info = OrderedDict()
            while Budget > 0:
                idx, jdx = np.random.choice(pool_set, size=2, replace=False)  # select two instances.
                Budget -= 1  # consume one budget.
                if y[idx] == y[jdx]:
                    continue  # no useful information is returned
                elif y[idx] > y[jdx]:
                    Relat_Info[(idx, jdx)] = ">"
                    Relat_Info[(jdx, idx)] = "<"
                elif y[idx] < y[jdx]:
                    Relat_Info[(idx, jdx)] = "<"
                    Relat_Info[(jdx, idx)] = ">"

            # -------------------training and testing-------------------------

            model = FARI()
            model.fit(Dist_Matrix=Dist_Matrix, y=y, abslabeled=labeled, Relat_info=Relat_Info)
            y_hat = model.predict(test_ids=test_idx)
            y_test = y[test_idx]

            MZE_list.append(1 - accuracy_score(y_true=y_test, y_pred=y_hat))
            MAE_list.append(mean_absolute_error(y_true=y_test, y_pred=y_hat))
            F1_list.append(f1_score(y_true=y_test, y_pred=y_hat,average="macro"))
            print("tril {} consume time::".format(SN), time()-S_Time)

        mean_MZE = np.mean(MZE_list)
        mean_MAE = np.mean(MAE_list)
        mean_F1 = np.mean(F1_list)
        std_MZE = np.std(MZE_list)
        std_MAE = np.std(MAE_list)
        std_F1 = np.std(F1_list)
        sheet_names = ["MZE_list","MAE_list","F1_list","MZE","MAE","F1"]
        workbook = xlwt.Workbook()
        for sn in sheet_names:
            sheet = workbook.add_sheet(sn)
            if sn == "MZE_list":
                for v, value in enumerate(MZE_list):
                    sheet.write(0,v,value)

            if sn == "MAE_list":
                for v, value in enumerate(MAE_list):
                    sheet.write(0,v,value)

            if sn == "F1_list":
                for v, value in enumerate(F1_list):
                    sheet.write(0,v,value)

            if sn == "MZE":
                sheet.write(0,0, mean_MZE)
                sheet.write(0,1, std_MZE)

            if sn == "MAE":
                sheet.write(0,0, mean_MAE)
                sheet.write(0,1, std_MAE)

            if sn == "F1":
                sheet.write(0,0, mean_F1)
                sheet.write(0,1, std_F1)
        save_path = Path(r"E:\EEEEE\ALResults\KNNARI_10K")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)








