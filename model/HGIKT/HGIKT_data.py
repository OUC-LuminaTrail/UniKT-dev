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
        hypergraph = self.generate_G_from_H(H, variable_weight=False)

        # 构建异构图
        hetero_graph = self.build_hetero_graph(
            [
                (
                    "question",
                    "has",
                    "skill",
                )
            ]
        )

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
        return train_dataloader, val_dataloader, hypergraph, hetero_graph
        
    def generate_G_from_H(self, H, variable_weight=False):  #从关联矩阵H生成超图G
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        import time
        print("Start constructing hypergraph...")
        t_start = time.time()

        # Convert to sparse matrix for efficiency
        if not sp.issparse(H):
            H = sp.csr_matrix(H)
        else:
            H = H.tocsr()

        n_edge = H.shape[1] # Number of columns of matrix = number of hyperedge 知识点的数量（列数）
        
        # the degree of the node
        DV = np.array(H.sum(axis=1)).flatten()
        
        # the degree of the hyperedge
        DE = np.array(H.sum(axis=0)).flatten()
        
        # Handle division by zero
        with np.errstate(divide='ignore'):
            invDE_val = np.power(DE, -1.0)
            DV2_val = np.power(DV, -0.5)
            
        invDE_val[np.isinf(invDE_val)] = 0
        DV2_val[np.isinf(DV2_val)] = 0
        
        invDE = sp.diags(invDE_val)
        DV2 = sp.diags(DV2_val)
        
        HT = H.T

        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            W = sp.eye(n_edge)
            print(f"Hypergraph construction finished in {time.time() - t_start:.2f}s")
            return DV2_H, W, invDE_HT_DV2
        else:
            # G = DV2 * H * W * invDE * HT * DV2
            # W is identity, so G = DV2 @ H @ invDE @ HT @ DV2
            G = DV2 @ H @ invDE @ HT @ DV2
            
            print(f"Hypergraph construction finished in {time.time() - t_start:.2f}s")
            G = self.sparse_mx_to_torch_sparse_tensor(G) #将矩阵G转为torch中的稀疏张量
            return G


    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor.把一个sparse matrix转为torch中的稀疏张量"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.tensor(np.stack([sparse_mx.row, sparse_mx.col]), dtype=torch.long)
        values = torch.tensor(sparse_mx.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, sparse_mx.shape).coalesce()
    