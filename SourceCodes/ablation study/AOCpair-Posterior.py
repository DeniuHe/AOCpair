import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from scipy.special import expit
from copy import deepcopy
from time import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv



class MREDKELM():
    def __init__(self):
        self.C = 10
        self.eX = None
        self.ey = None

    def fit(self, X, y, nSample, nDim, nClass, labels, train_ids, Xin):
        self.X = X
        self.y = y
        self.nSample = nSample
        self.nDim = nDim
        self.nClass = nClass
        self.nTheta = nClass -1
        self.labels = labels
        self.train_ids = train_ids
        self.Xin = Xin
        self.extend_part = np.eye(self.nClass-1)
        self.label_dict = self.Get_binary_label()
        self.eX, self.ey = self.train_set_construct()
        self.gram_train = self.get_gram_train()
        self.beta = inv(0.1*np.eye(self.gram_train.shape[0]) + self.gram_train) @ self.ey
        return self


    def Get_binary_label(self):
        label_dict = OrderedDict()
        for i, lab in enumerate(self.labels):
            tmp_label = np.ones(self.nClass-1)
            for k in range(self.nClass-1):
                if i <= k:
                    tmp_label[k] = -5
                else:
                    tmp_label[k] = 5
            label_dict[lab] = tmp_label
        return label_dict

    def train_set_construct(self):
        eX = []
        ey = []
        for idx in self.train_ids:
            eXi = np.hstack((np.tile(self.X[idx], (self.nTheta, 1)), self.extend_part))
            for k, pad in enumerate(self.labels[:-1]):
                if self.Xin[idx].inter[-1] <= pad:
                    eX.append(eXi[k,:])
                    ey.append(-1)
                elif self.Xin[idx].inter[0] > pad:
                    eX.append(eXi[k,:])
                    ey.append(1)
        eX, ey = np.asarray(eX), np.asarray(ey)
        return eX, ey

    def get_gram_train(self):
        gram_train_1 = -pairwise_distances(X=self.eX[:,:self.nDim],Y=self.eX[:,:self.nDim],metric="euclidean")
        gram_train_2 = self.eX[:,self.nDim:] @ self.eX[:,self.nDim:].T
        gram_train = gram_train_1 + gram_train_2
        return gram_train

    def test_set_construct(self, X_test):
        nTest = X_test.shape[0]
        eX = np.zeros((nTest * self.nTheta, self.nDim + self.nTheta))
        for i in range(nTest):
            eXi = np.hstack((np.tile(X_test[i],(self.nTheta,1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
        return eX

    def get_gram_test(self, eX_test):
        gram_test_1 = -pairwise_distances(X=eX_test[:,:self.nDim],Y=self.eX[:,:self.nDim],metric="euclidean")
        gram_test_2 = eX_test[:,self.nDim:] @ self.eX[:,self.nDim:].T
        gram_test = gram_test_1 + gram_test_2
        return gram_test

    def predict(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        gram_test = self.get_gram_test(eX_test)
        y_extend = np.sign(gram_test @ self.beta)
        y_tmp = y_extend.reshape(nTest,self.nTheta)
        y_pred = np.sum(y_tmp > 0, axis=1)
        return y_pred

    def distant_to_theta(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        gram_test = self.get_gram_test(eX_test)
        dist_tmp = gram_test @ self.beta

        dist_matrix = -dist_tmp.reshape(nTest, self.nTheta)
        return dist_matrix

    def predict_proba(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        gram_test = self.get_gram_test(eX_test)
        dist_tmp = gram_test @ self.beta
        dist_matrix = -dist_tmp.reshape(nTest, self.nTheta) * 10
        accumulative_proba = expit(dist_matrix)
        prob = np.pad(
            accumulative_proba,
            pad_width=((0, 0), (1, 1)),
            mode='constant',
            constant_values=(0, 1))
        prob = np.diff(prob)
        return prob


class Struct(object):
    def __init__(self, idx):
        self.inter = None
        self.prob = OrderedDict()
        self.evalab = None



class MS_Median():
    def __init__(self, X, y, labeled, budget, X_test, y_test):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.X_test = X_test
        self.y_test = y_test
        self.abslabeled = list(deepcopy(labeled))
        self.model = MREDKELM()
        self.n_theta = [i for i in range(self.nClass - 1)]
        self.theta = None
        self.unlabeled = self.init_unlabeled()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.recordNote = [i for i in np.arange(1, self.budget + 1, 1)]
        self.initNum = len(labeled)

        # -------------------------------
        self.tmp_selected = []
        self.intlabeled = []
        self.Xin = self.init_pool_information()
        self.already = []
        # -------------------------------
        self.MZElist = []
        self.MAElist = []
        self.F1list = []
        self.ALC_MZE = 0.0
        self.ALC_MAE = 0.0
        self.ALC_F1 = 0.0


    def init_unlabeled(self):
        unlabeled = list(range(self.nSample))
        for idx in self.abslabeled:
            unlabeled.remove(idx)
        return unlabeled

    def init_pool_information(self):
        Xin = OrderedDict()
        for idx in self.abslabeled:
            Xin[idx] = Struct(idx=idx)
            Xin[idx].inter = [self.y[idx], self.y[idx]]
        for idx in self.unlabeled:
            Xin[idx] = Struct(idx=idx)
            Xin[idx].inter = deepcopy(self.labels)
        return Xin


    def reTrain(self):
        train_ids = self.abslabeled + self.intlabeled
        self.model.fit(X=self.X, y=self.y, nSample=self.nSample, nDim=self.nDim, nClass=self.nClass,
                       labels=self.labels, train_ids=train_ids, Xin=self.Xin)

    def calculate_unlabeled_intlabeled_proba(self):
        if self.unlabeled:
            prob_matrix = self.model.predict_proba(self.X[self.unlabeled])
            for i, idx in enumerate(self.unlabeled):
                self.Xin[idx].prob = OrderedDict()
                for r, lab in enumerate(self.labels):
                    self.Xin[idx].prob[lab] = prob_matrix[i,r]
        if self.intlabeled:
            dist_matrix = self.model.distant_to_theta(self.X[self.intlabeled])
            for i, idx in enumerate(self.intlabeled):
                class_interval = deepcopy(self.Xin[idx].inter)
                lab_index_list = []
                for lab in class_interval[:-1]:
                    lab_index_list.append(self.labels.index(lab))
                # TODO 验证
                dist_tmp = dist_matrix[i, lab_index_list]
                accumulative_proba = expit(dist_tmp)
                prob = np.pad(
                    [accumulative_proba],
                    pad_width=((0, 0), (1, 1)),
                    mode='constant',
                    constant_values=(0, 1))
                prob_list = np.diff(prob)[0]
                prob_dict = OrderedDict()
                for r, lab in enumerate(class_interval):
                    prob_dict[lab] = prob_list[r]
                self.Xin[idx].prob = prob_dict


    def select_unlabeled_point(self):
        abs_dist_mat = abs(self.model.distant_to_theta(self.X[self.unlabeled]))
        min_dist_list = np.min(abs_dist_mat, axis=1)
        ordidx = np.argsort(min_dist_list)
        tar_idx = self.unlabeled[ordidx[0]]
        self.unlabeled.remove(tar_idx)
        self.intlabeled.append(tar_idx)
        print("选择一个无标记样本:{}".format(tar_idx), " 现有区间样本{}".format(self.intlabeled))


    def select_labeled_point(self):
        '''
        select labeled instance based on posterior
        '''
        for idx in self.intlabeled:
            self.Xin[idx].evalab = max(self.Xin[idx].prob, key=self.Xin[idx].prob.get)

    def evaluation(self):
        note = self.budget - self.budgetLeft
        y_pred = self.model.predict(X=self.X_test)
        if note not in self.already:
            self.already.append(note)
            self.MZElist.append(1-accuracy_score(y_true=self.y_test, y_pred=y_pred))
            self.MAElist.append(mean_absolute_error(y_true=self.y_test, y_pred=y_pred))
            self.F1list.append(f1_score(y_true=self.y_test, y_pred=y_pred, average='macro'))
        else:
            self.MZElist[-1] = 1-accuracy_score(y_true=self.y_test, y_pred=y_pred)
            self.MAElist[-1] = mean_absolute_error(y_true=self.y_test, y_pred=y_pred)
            self.F1list[-1] = f1_score(y_true=self.y_test, y_pred=y_pred, average='macro')




    def query_reasoning(self):
        if self.nClass <= 3:
            for idx in self.tmp_selected:
                if self.budgetLeft <= 0:
                    break
                else:
                    self.budgetLeft -= 1
                    self.abslabeled.append(idx)
                    self.Xin[idx].inter = [self.y[idx], self.y[idx]]
                    self.evaluation()
        else:
            intlabeled = deepcopy(self.intlabeled)
            for idx in intlabeled:
                if self.budgetLeft <= 0:
                    print("跳出")
                    break
                else:
                    self.budgetLeft -= 1
                    print("推理{}中的{}".format(intlabeled,idx)," 访问预算减小为{}".format(self.budgetLeft))
                    if self.y[idx] == self.Xin[idx].evalab:
                        self.abslabeled.append(idx)
                        self.Xin[idx].inter = [self.y[idx], self.y[idx]]
                        self.intlabeled.remove(idx)
                    elif self.y[idx] < self.Xin[idx].evalab:
                        if self.Xin[idx].evalab - self.Xin[idx].inter[0] == 1:
                            self.abslabeled.append(idx)
                            self.Xin[idx].inter = [self.y[idx], self.y[idx]]
                            self.intlabeled.remove(idx)
                        elif self.Xin[idx].evalab - self.Xin[idx].inter[0] > 1:
                            self.Xin[idx].inter = [_ for _ in np.arange(self.Xin[idx].inter[0], self.Xin[idx].evalab, 1)]
                    elif self.y[idx] > self.Xin[idx].evalab:
                        if self.Xin[idx].inter[-1] - self.Xin[idx].evalab == 1:
                            self.abslabeled.append(idx)
                            self.Xin[idx].inter = [self.y[idx],self.y[idx]]
                            self.intlabeled.remove(idx)
                        elif self.Xin[idx].inter[-1] - self.Xin[idx].evalab > 1:
                            self.Xin[idx].inter = [_ for _ in np.arange(self.Xin[idx].evalab+1, self.Xin[idx].inter[-1] + 1, 1)]

                self.reTrain()
                self.evaluation()
                self.calculate_unlabeled_intlabeled_proba()
                print("推理后，区间样本{}的区间为{},概率为{}".format(idx,self.Xin[idx].inter, self.Xin[idx].prob))



    def start(self):
        print("The first time to conduct Function reTrain")
        self.reTrain()
        self.calculate_unlabeled_intlabeled_proba()
        print("   ")
        while self.budgetLeft > 0:
            print("   ")
            print("##---%%---%%---%%---##")
            print("绝对标记样本：{} -> {}".format(self.abslabeled, self.y[self.abslabeled]))
            print("区间标记样本：{} -> {}".format(self.intlabeled, self.y[self.intlabeled]))
            print("##---%%---%%---%%---##")
            print("    ")

            self.select_unlabeled_point()
            self.select_labeled_point()
            self.query_reasoning()

        self.ALC_MZE = sum(self.MZElist)
        self.ALC_MAE = sum(self.MAElist)
        self.ALC_F1 = sum(self.F1list)








if __name__ == '__main__':
    p = Path("E:\EEEEE\Dataset")

    names_list = ["SWD","Car","Automobile","Cleveland","Housing-5bin","Stock-5bin","Computer-5bin","Winequality-red","Obesity","Housing-10bin","Stock-10bin","Computer-10bin"]


    class results():
        def __init__(self):
            self.MZEList = []
            self.MAEList = []
            self.F1List = []
            self.ALC_MZE = []
            self.ALC_MAE = []
            self.ALC_F1 = []

    class stores():
        def __init__(self):
            self.MZEList_mean = []
            self.MZEList_std = []
            # -----------------
            self.MAEList_mean = []
            self.MAEList_std = []
            # -----------------
            self.F1List_mean = []
            self.F1List_std = []
            # -----------------
            # -----------------
            self.ALC_MZE_mean = []
            self.ALC_MZE_std = []
            # -----------------
            self.ALC_MAE_mean = []
            self.ALC_MAE_std = []
            # -----------------
            self.ALC_F1_mean = []
            self.ALC_F1_std = []
            # -----------------
            self.ALC_MZE_list = []
            self.ALC_MAE_list = []
            self.ALC_F1_list = []

    method = "Post"
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
        Budget = 20 * nClass

        """--------read the partitions--------"""
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)

        # --------------------------------------
        RESULT = results()
        STORE = stores()
        # --------------------------------------

        workbook = xlwt.Workbook()
        count = 0
        for SN in book_partition.sheet_names():
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

            model = MS_Median(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
            model.start()
            RESULT.MZEList.append(model.MZElist)
            RESULT.MAEList.append(model.MAElist)
            RESULT.F1List.append(model.F1list)
            RESULT.ALC_MZE.append(model.ALC_MZE)
            RESULT.ALC_MAE.append(model.ALC_MAE)
            RESULT.ALC_F1.append(model.ALC_F1)

        STORE.MZEList_mean = np.mean(RESULT.MZEList, axis=0)
        STORE.MZEList_std = np.std(RESULT.MZEList, axis=0)
        STORE.MAEList_mean = np.mean(RESULT.MAEList, axis=0)
        STORE.MAEList_std = np.std(RESULT.MAEList, axis=0)
        STORE.F1List_mean = np.mean(RESULT.F1List, axis=0)
        STORE.F1List_std = np.std(RESULT.F1List, axis=0)
        STORE.ALC_MZE_mean = np.mean(RESULT.ALC_MZE)
        STORE.ALC_MZE_std = np.std(RESULT.ALC_MZE)
        STORE.ALC_MAE_mean = np.mean(RESULT.ALC_MAE)
        STORE.ALC_MAE_std = np.std(RESULT.ALC_MAE)
        STORE.ALC_F1_mean = np.mean(RESULT.ALC_F1)
        STORE.ALC_F1_std = np.std(RESULT.ALC_F1)
        STORE.ALC_MZE_list = RESULT.ALC_MZE
        STORE.ALC_MAE_list = RESULT.ALC_MAE
        STORE.ALC_F1_list = RESULT.ALC_F1

        sheet_names = ["MZE_mean", "MZE_std", "MAE_mean", "MAE_std", "F1_mean", "F1_std",
                       "ALC_MZE_list","ALC_MAE_list","ALC_F1_list",
                       "ALC_MZE", "ALC_MAE", "ALC_F1"]
        workbook = xlwt.Workbook()
        for sn in sheet_names:
            print("sn::",sn)
            sheet = workbook.add_sheet(sn)
            n_col = len(STORE.MZEList_mean)
            if sn == "MZE_mean":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MZEList_mean[j - 1])
            elif sn == "MZE_std":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MZEList_std[j - 1])
            elif sn == "MAE_mean":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MAEList_mean[j - 1])
            elif sn == "MAE_std":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.MAEList_std[j - 1])
            elif sn == "F1_mean":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.F1List_mean[j - 1])
            elif sn == "F1_std":
                sheet.write(0, 0, method)
                for j in range(1,n_col + 1):
                    sheet.write(0,j,STORE.F1List_std[j - 1])

            # ---------------------------------------------------
            elif sn == "ALC_MZE_list":
                sheet.write(0, 0, method)
                for j in range(1,len(STORE.ALC_MZE_list) + 1):
                    sheet.write(0,j,STORE.ALC_MZE_list[j - 1])
            elif sn == "ALC_MAE_list":
                sheet.write(0, 0, method)
                for j in range(1,len(STORE.ALC_MAE_list) + 1):
                    sheet.write(0,j,STORE.ALC_MAE_list[j - 1])
            elif sn == "ALC_F1_list":
                sheet.write(0, 0, method)
                for j in range(1,len(STORE.ALC_F1_list) + 1):
                    sheet.write(0,j,STORE.ALC_F1_list[j - 1])
            # -----------------
            elif sn == "ALC_MZE":
                sheet.write(0, 0, method)
                sheet.write(0, 1, STORE.ALC_MZE_mean)
                sheet.write(0, 2, STORE.ALC_MZE_std)
            elif sn == "ALC_MAE":
                sheet.write(0, 0, method)
                sheet.write(0, 1, STORE.ALC_MAE_mean)
                sheet.write(0, 2, STORE.ALC_MAE_std)
            elif sn == "ALC_F1":
                sheet.write(0, 0, method)
                sheet.write(0, 1, STORE.ALC_F1_mean)
                sheet.write(0, 2, STORE.ALC_F1_std)

        save_path = Path(r"E:\EEEEE\ALResults\MS-Posterior")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)


