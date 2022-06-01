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


class svlorr():
    def __init__(self):
        self.X_train = None
        self.y_train = None

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
        self._w = np.zeros(self.nDim)
        self.epoch= 500
        self.lr = 0.01
        self.C = 1
        self.Z = self.get_z()
        self.relativeSub = self.Get_relative_information()
        self.update()
        self.boundary = self.get_boundary()

    def get_z(self):
        Z = np.zeros((self.nSample, self.nSample))
        for i in range(self.nSample-1):
            for j in range(i, self.nSample):
                Z[i,j] = np.sign(self.y_train[i] - self.y_train[j])
        return Z

    def Get_relative_information(self):
        interSub = []
        for ele in self.selected_pair:
            interSub.append(self.X_pool[ele[0]] - self.X_pool[ele[1]])
        return interSub

    def update(self):
        for e in range(self.epoch):
            G = 0.0
            for i in range(self.nSample-1):
                for j in range(i,self.nSample):
                    # --------------计算梯度------------------
                    zSubX = self.Z[i,j] * (self.X_train[i] - self.X_train[j])
                    wzSubX = self._w @ zSubX
                    if 1 > wzSubX:
                        G += - zSubX
            for subInter in self.relativeSub:
                if self._w @ subInter < 1:
                    G += - subInter
            G = self.C * G + self._w
            self._w -= self.lr * G


    def get_boundary(self):
        lab_dict = OrderedDict()
        for lab in self.labels:
            lab_dict[lab] = []
        for idx in range(self.nSample):
            lab_dict[self.y_train[idx]].append(idx)

        Theta = np.zeros(self.nClass-1)
        for k in range(self.nClass-1):
            min_value = np.inf
            tar_ele = None
            for ele in product(lab_dict[k],lab_dict[k+1]):
                SubXw = self._w @ (self.X_train[ele[0]] - self.X_train[ele[1]])
                if min_value > SubXw:
                    tar_ele = ele
                    min_value = SubXw
            Theta[k] = self._w @ (self.X_train[tar_ele[0]] + self.X_train[tar_ele[1]]) / 2
        return Theta


    def predict(self,X):
        tmp = X @ self._w - self.boundary[:,None]
        y_pred = np.sum(tmp > 0, axis=0).astype(np.int32)
        return y_pred







if __name__ == '__main__':
    p = Path("E:\EEEEE\Dataset")

    names_list = ["SWD","Car","Automobile","Cleveland","Housing-5bin","Stock-5bin","Computer-5bin","Winequality-red","Obesity","Housing-10bin","Stock-10bin","Computer-10bin"]

    class results():
        def __init__(self):
            self.MZEList = []
            self.MAEList = []
            self.F1List = []

    method = "SVLORR"
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
            Budget = 20 * nClass
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

            model = svlorr()
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
        save_path = Path(r"E:\EEEEE\ALResults\SVLORR_20K")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)

