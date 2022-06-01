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



class MS_ECM():
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
        # print("选择一个无标记样本:{}".format(tar_idx), " 现有区间样本{}".format(self.intlabeled))


    def select_labeled_point(self):
        '''
        select labeled instance based on Expected Cost Minimization
        '''
        for idx in self.intlabeled:
            if len(self.Xin[idx].inter) <= 3:
                self.Xin[idx].evalab = self.Xin[idx].inter[1]
            else:
                ECM = OrderedDict()
                for lab in self.Xin[idx].inter[1:-1]:
                    same_cost = self.Xin[idx].prob[lab] * 1
                    left_cost = 0.0
                    right_cost = 0.0
                    left_price = np.floor(np.log2(lab - self.Xin[idx].inter[0]))
                    right_price = np.floor(np.log2(self.Xin[idx].inter[-1] - lab))
                    for ele, pro in self.Xin[idx].prob.items():
                        if ele < lab:
                            left_cost += pro * (1 + left_price)
                        elif ele > lab:
                            right_cost += pro * (1 + right_price)
                    ECM[lab] = same_cost + left_cost + right_cost
                tar_lab_cost = min(ECM, key=ECM.get)
                self.Xin[idx].evalab = tar_lab_cost


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
                    # print("跳出")
                    break
                else:
                    self.budgetLeft -= 1
                    # print("推理{}中的{}".format(intlabeled,idx)," 访问预算减小为{}".format(self.budgetLeft))
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
                # print("推理后，区间样本{}的区间为{},概率为{}".format(idx,self.Xin[idx].inter, self.Xin[idx].prob))



    def start(self):
        # print("The first time to conduct Function reTrain")
        self.reTrain()
        self.calculate_unlabeled_intlabeled_proba()
        # print("   ")
        while self.budgetLeft > 0:
            # print("   ")
            # print("##---%%---%%---%%---##")
            # print("绝对标记样本：{} -> {}".format(self.abslabeled, self.y[self.abslabeled]))
            # print("区间标记样本：{} -> {}".format(self.intlabeled, self.y[self.intlabeled]))
            # print("##---%%---%%---%%---##")
            # print("    ")

            self.select_unlabeled_point()
            self.select_labeled_point()
            self.query_reasoning()

        self.ALC_MZE = sum(self.MZElist)
        self.ALC_MAE = sum(self.MAElist)
        self.ALC_F1 = sum(self.F1list)



if __name__ == '__main__':
    p = Path("E:\EEEEE\Dataset")

    names_list = ["SWD","Car","Automobile","Cleveland","Housing-5bin","Stock-5bin","Computer-5bin","Winequality-red","Obesity","Housing-10bin","Stock-10bin","Computer-10bin"]

    method = "MS-ECM"
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
        Budget = 10 * nClass

        """--------read the partitions--------"""
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)

        # --------------------------------------

        # --------------------------------------

        workbook = xlwt.Workbook()
        count = 0
        MZE_list = []
        MAE_list = []
        F1_list = []
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

            model = MS_ECM(X=X_train, y=y_train, labeled=labeled, budget=Budget, X_test=X_test, y_test=y_test)
            model.start()


            MZE_list.append(model.MZElist[-1])
            MAE_list.append(model.MAElist[-1])
            F1_list.append(model.F1list[-1])
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
        save_path = Path(r"E:\EEEEE\ALResults\ECM_10K")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)
