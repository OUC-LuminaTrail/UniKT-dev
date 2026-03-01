from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import BaseModelData


class SkillModelData(BaseModelData):
    """
    技能序列数据基类

    用于构建基于技能（skill/concept）的知识追踪模型数据
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    def build_sequence_data(self, max_seq_len: int, min_seq_len: int):
        r"""
        构建用户答题序列，将问题ID映射到技能ID，并展开多知识点

        参数:
            max_seq_len: 最大序列长度
            min_seq_len: 最小序列长度

        返回:
            user_sequence: 用户技能ID序列，shape为(num_users, max_seq_len)
            user_response: 用户响应序列，shape为(num_users, max_seq_len)
            user_mask: 用户掩码序列，shape为(num_users, max_seq_len)
            user_id_sequence: 用户ID序列，shape为(num_users, max_seq_len)
        """
        import numpy as np
        from tqdm import tqdm

        data = self.data_src.get_sequence_data()
        question_data = self.data_src.get_question_data()
        num_users = self.data_src.get_metadata("num_users")

        # 构建问题ID到技能ID列表的映射
        # 在数据预处理中，question_data已经将多知识点展开为多个技能ID
        # 例如：question_id=1, skill=10; question_id=1, skill=20
        question_to_skills = {}
        for row in question_data.itertuples():
            qid = row.question
            sid = row.skill
            if qid not in question_to_skills:
                question_to_skills[qid] = []
            question_to_skills[qid].append(sid)

        # 初始化序列数组
        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)

        # 用户序列长度计数
        num_sequence = [0] * num_users

        # 按用户分组构建序列，展开多知识点
        for row in tqdm(
            data.itertuples(), total=data.shape[0], desc="Building skill sequences"
        ):
            user_idx = row.user
            question_idx = row.question
            label = row.label

            # 获取该问题对应的所有技能
            skills = question_to_skills.get(question_idx, [0])

            # 展开多知识点：一个问题对应多个技能时，展开成多个独立的交互
            for skill_idx in skills:
                # 如果当前用户的序列长度未达到最大长度，则添加数据
                if num_sequence[user_idx] < max_seq_len:
                    user_sequence[user_idx, num_sequence[user_idx]] = skill_idx
                    user_id_sequence[user_idx, num_sequence[user_idx]] = user_idx
                    user_response[user_idx, num_sequence[user_idx]] = label
                    user_mask[user_idx, num_sequence[user_idx]] = 1
                    num_sequence[user_idx] += 1
                else:
                    # 如果序列已满，跳过剩余的技能
                    break

        self.logger.debug(
            f"Built skill sequences for {num_users} users, max_len={max_seq_len}"
        )
        self.logger.debug(
            f"Multi-skill expansion applied: {len(question_to_skills)} questions mapped to skills"
        )

        return user_sequence, user_response, user_mask, user_id_sequence
