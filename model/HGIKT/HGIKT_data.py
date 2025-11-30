import torch
import numpy as np
import scipy.sparse as sp
from utility.data_process.data_utility import DataSource, ModelData
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing_extensions import override


class HGIKTDataset(Dataset):
    def __init__(self, sequences, responses, masks):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class HGIKTModelData(ModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        r"""
        准备HGIKT模型所需的数据
        """
        fold_idx = args.fold if args.fold > 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        min_seq_len = self.data_src.get_metadata("min_seq_len")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            max_seq_len, min_seq_len
        )

        # 构建题目-知识点二值关联矩阵（题目行 - 知识点列）
        # 使用工具库的 build_data_matrix 直接从已处理数据生成二值矩阵，值为 0/1
        H = self.build_data_matrix(("question", "has", "skill"), value_type="binary")

        # 构建超图
        graph = self.generate_G_from_H(H, variable_weight=False)

        # 划分训练集和验证集
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            print(f"Using K-fold cross-validation: fold {fold_idx}/{kfold_n_splits}")
            train_data, val_data = self.get_kfold_split_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            train_data, val_data = self.split_data(
                user_sequence, user_response, user_mask
            )

        # 构建模型数据集
        train_dataset = HGIKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = HGIKTDataset(val_data[0], val_data[1], val_data[2])

        # 构建数据加载器
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        return train_dataloader, val_dataloader, graph
        
    def generate_G_from_H(self, H, variable_weight=False):  #从关联矩阵H生成超图G
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        H = np.array(H)
        n_edge = H.shape[1] # Number of columns of matrix = number of hyperedge 知识点的数量（列数）
        # the weight of the hyperedge
        W = np.ones(n_edge)  #获得一个全1的矩阵
        # the degree of the node
        DV = np.sum(H *W, axis=1)   #节点度（题目关联多少概念）
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)     #超边度（知识点被多少题目关联）
        invDE = np.asmatrix(np.diag(np.power(DE, float(-1))))
        DV2 = np.asmatrix(np.diag(np.power(DV, -0.5)))
        W = np.asmatrix(np.diag(W))
        H = np.asmatrix(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2  #公式G = Dv^-1/2 * H * W * De^-1 * H.T * Dv^-1/2 归一化矩阵？
            G = self.sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(G)) #将矩阵G转为torch中的稀疏张量
            # G = torch.Tensor(G)
            return G


    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor.把一个sparse matrix转为torch中的稀疏张量"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape) 