import torch
import numpy as np
from utility.data_process.data_utility import DataSource
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class GIKTDataset(Dataset):
    def __init__(self, sequences, responses, masks, graph):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.graph = graph

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


def build_hetro_graph(data_src: DataSource):
    from tqdm import tqdm
    # 构建异构问题-技能图
    data = data_src.get_processed_data()
    graph = HeteroData()
    
    # 获取问题、用户和技能数量
    num_question = data["question_id"].nunique()
    num_skill = data["skill_id"].nunique()
    
    # 问题-技能边列表
    qs_edge_list = set()
    
    # 遍历数据，构建问题-技能边
    for row in tqdm(
        data[["question_id", "skill_id"]].itertuples(index=False),
        total=data.shape[0],
        desc="Building question-skill edges",
    ):
        q_id = row.question_id
        s_id = row.skill_id
        qs_edge_list.add((q_id, s_id))
    
    # 转换为列表并构建边索引张量
    qs_edge_list = list(qs_edge_list)
    qs_edge_index = np.array(qs_edge_list, dtype=np.int64).T
    qs_edge_index = torch.tensor(qs_edge_index, dtype=torch.long).contiguous()
    
    # 设置节点数量
    graph["question"].num_nodes = num_question
    graph["skill"].num_nodes = num_skill
    
    # 设置节点特征（使用节点ID作为特征）
    graph["question"].x = torch.arange(num_question).view(-1, 1).float()
    graph["skill"].x = torch.arange(num_skill).view(-1, 1).float()
    
    # 添加问题-技能边
    graph["question", "has_skill", "skill"].edge_index = qs_edge_index
    
    # 添加反向边，构建无向图
    graph = ToUndirected()(graph)
    
    return graph


def build_sequence_data(data_src: DataSource, max_seq_len: int, min_seq_len: int):
    from tqdm import tqdm

    data = data_src.get_processed_data()
    num_users = data_src.get_metadata("num_users")

    # 构建用户答题序列
    user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
    # 用户作答正确与否序列
    user_response = np.zeros((num_users, max_seq_len), dtype=int)
    # 序列掩码，用于区分是否存在作答数据
    user_mask = np.zeros((num_users, max_seq_len), dtype=int)
    # 用户序列长度计数器，用于索引
    num_sequence = [0] * num_users

    for row in tqdm(
        data.itertuples(), total=data.shape[0], desc="Building user sequences"
    ):
        # 获取用户ID、问题ID和作答正确与否
        user_idx = row.user_id
        question_idx = row.question_id
        label = row.label
        # 如果当前用户的序列长度未达到最大长度，则添加数据
        if num_sequence[user_idx] < max_seq_len:
            user_sequence[user_idx, num_sequence[user_idx]] = question_idx
            user_response[user_idx, num_sequence[user_idx]] = label
            user_mask[user_idx, num_sequence[user_idx]] = 1
            # 自增对应的用户序列长度
            num_sequence[user_idx] += 1

    return user_sequence, user_response, user_mask


def split_data(sequences, responses, masks, val_ratio=0.2):
    from tqdm import tqdm
    num_users, max_seq_len = sequences.shape

    # 计算每个用户实际的序列长度
    seq_lengths = np.sum(masks, axis=1).astype(int)
    
    # 计算分割点：每个用户的验证集从哪个时间步开始
    split_points = np.ceil(seq_lengths * (1 - val_ratio)).astype(int)
    train_sequences = sequences.copy()
    train_responses = responses.copy()
    train_masks = masks.copy()
    
    val_sequences = sequences.copy()
    val_responses = responses.copy()
    val_masks = masks.copy()
    
    # 对每个用户，按时间步分割
    for user_idx in tqdm(range(num_users), desc="Split dataset"):
        split_point = split_points[user_idx]
        actual_length = seq_lengths[user_idx]
        
        if actual_length == 0:
            continue

        # 训练集：前 split_point 个时间步
        train_masks[user_idx, split_point:] = 0
        train_responses[user_idx, split_point:] = 0
        
        # 验证集：后面的时间步
        val_masks[user_idx, :split_point] = 0
        val_responses[user_idx, :split_point] = 0
    
    train_data = (train_sequences, train_responses, train_masks)
    val_data = (val_sequences, val_responses, val_masks)
    
    return train_data, val_data


def build_data(args, data_src: DataSource):
    # 构建用户答题序列
    user_sequence, user_response, user_mask = build_sequence_data(
        data_src, data_src.get_metadata("max_seq_len"), data_src.get_metadata("min_seq_len")
    )

    # 构建异构图
    graph = build_hetro_graph(data_src)

    # 划分训练集和验证集
    train_data, val_data = split_data(user_sequence, user_response, user_mask)
    # 构建模型数据集
    train_dataset = GIKTDataset(train_data[0], train_data[1], train_data[2], graph)
    val_dataset = GIKTDataset(val_data[0], val_data[1], val_data[2], graph)
    # 构建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, graph
