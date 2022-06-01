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
from scipy.special import expit
from itertools import product


class ldlorr():
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y, abslabeled, selected_pair):
        self.X_pool = X
        self.y_pool = y
        self.X_train = np.asarray(X[abslabeled], dtype=np.float64)
        self.y_train = np.asarray(y[abslabeled], dtype=np.int32)
        self.abslabeled = abslabeled
        self.selected_pair = selected_pair
        self.nSample, self.nDim = self.X_train.shape
        self.labels = list(np.sort(np.unique(y)))
        self.nClass = len(self.labels)
        self.epoch= 500
        self.lr = 0.01
        self.C = 1
        self.nEachClass = []
        self.ClassIndex = OrderedDict()
        self.Mean = self.get_Mean()
        self.subMean = self.get_subMean()
        self.Sw = self.get_Sw()
        self.invSw = np.linalg.inv(self.Sw)
        self.w = np.random.random(self.nDim)
        self.relativeSub = self.Get_relative_information()
        self.update()
        self.boundary = self.get_boundary()


    def get_Mean(self):
        Mean = np.zeros((self.nClass,self.nDim))
        for i, lab in enumerate(self.labels):
            idx_list = np.where(self.y_train == lab)[0]
            Mean[i] = np.mean(self.X_train[idx_list],axis=0)
            self.nEachClass.append(np.size(idx_list))
            self.ClassIndex[i] = idx_list
        return Mean

    def get_Sw(self):
        Sw = np.zeros((self.nDim, self.nDim))
        for i, lab in enumerate(self.labels):
            Xi = self.X_train[self.ClassIndex[i]] - self.Mean[i]
            Sw += (self.nEachClass[i]/self.nSample) * Xi.T @ Xi
        Sw = Sw + 0.0001 * np.identity(self.nDim)
        return Sw

    def get_subMean(self):
        subMean = np.ones((self.nClass-1, self.nDim))
        for k in range(self.nClass-1):
            subMean[k] = self.Mean[k+1] - self.Mean[k]
        return subMean

    def get_boundary(self):
        sumMean = np.zeros((self.nClass-1, self.nDim))
        for k in range(self.nClass-1):
            sumMean[k] = 1 / (self.nEachClass[k] + self.nEachClass[k+1]) * (self.nEachClass[k] * self.Mean[k] + self.nEachClass[k+1] * self.Mean[k+1])
        boundary = sumMean @ self.w
        return boundary

    def predict(self,X):
        tmp = X @ self.w - self.boundary[:,None]
        # print("tmp::",tmp.shape)
        y_pred = np.sum(tmp > 0, axis=0).astype(np.int32)
        return y_pred


    def Get_relative_information(self):
        interSub = []
        for ele in self.selected_pair:
            interSub.append(self.X_pool[ele[0]] - self.X_pool[ele[1]])
        return interSub

    def update(self):
        for e in range(self.epoch):
            G = 2 * self.Sw @ self.w
            for k in range(self.nClass-1):
                if self.w @ self.subMean[k] < 1:
                    G += -self.C * self.subMean[k]
            for subInter in self.relativeSub:
                if self.w @ subInter < 1:
                    G += -self.C * subInter
            self.w -= self.lr * G





if __name__ == '__main__':
    p = Path("E:\EEEEE\Dataset")

    names_list = ["SWD","Car","Automobile","Cleveland","Housing-5bin","Stock-5bin","Computer-5bin","Winequality-red","Obesity","Housing-10bin","Stock-10bin","Computer-10bin"]

    class results():
        def __init__(self):
            self.MZEList = []
            self.MAEList = []
            self.F1List = []

    method = "LDLORR"
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

            X_train = X[train_idx]
            y_train = y[train_idx].astype(np.int32)
            X_test = X[test_idx]
            y_test = y[test_idx]



            # ---------Get the Relative information for KNNARI model-------------
            # ----initialize the poo_set-----
            pool_set = list(range(len(train_idx)))
            for idx in labeled:
                pool_set.remove(idx)
            # ---select the query pairs-----
            Relat_Info = []
            while Budget > 0:
                idx, jdx = np.random.choice(pool_set, size=2, replace=False)  # select two instances.
                Budget -= 1  # consume one budget.
                if y_train[idx] == y_train[jdx]:
                    continue  # no useful information is returned
                elif y_train[idx] > y_train[jdx]:
                    Relat_Info.append((idx, jdx))
                elif y_train[idx] < y_train[jdx]:
                    Relat_Info.append((jdx, idx))


            # -------------------training and testing-------------------------

            model = ldlorr()
            model.fit(X=X_train, y=y_train, abslabeled=labeled,selected_pair=Relat_Info)
            y_hat = model.predict(X=X_test)

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
        save_path = Path(r"E:\EEEEE\ALResults\LDLORR_10K")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)

