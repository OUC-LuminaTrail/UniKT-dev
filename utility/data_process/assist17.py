import os
import pandas as pd
from typing_extensions import override
from .data_utility import DataSource


class Assistments2017Data(DataSource):
    """
    Assistments2017数据集处理类
    """

    def __init__(self, args):
        super().__init__(
            dataset="assistments17",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments17.zip",
            seed=args.seed,
        )
        self.args = args
        # 原始数据文件路径
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "anonymized_full_release_competition_dataset.csv"
        )

    @override
    def load_src_data(self):
        r"""
        加载原始数据
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        print("Loading raw data from:", self.raw_data_path)
        self.raw_data = pd.read_csv(
            self.raw_data_path, encoding="latin1", low_memory=False
        )

    @override
    def clear_data(self):
        print("Processing Data...")
        if self.raw_data is None:
            try:
                self.load_src_data()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Raw data not found. Please fetch the data first."
                )
        data = self.raw_data.drop(
            columns=[
                # "studentId",
                "MiddleSchoolId",
                "InferredGender",
                "SY ASSISTments Usage",
                "AveKnow",
                "AveCarelessness",
                "AveCorrect",
                "NumActions",
                "AveResBored",
                "AveResEngcon",
                "AveResConf",
                "AveResFrust",
                "AveResOfftask",
                "AveResGaming",
                "action_num",
                # "skill",
                # "problemId",
                "problemType",
                "assignmentId",
                "assistmentId",
                # "startTime",
                "endTime",
                "timeTaken",
                # "correct",
                # "original",
                "hint",
                "hintCount",
                "hintTotal",
                "scaffold",
                "bottomHint",
                "attemptCount",
                "frIsHelpRequest",
                "frPast5HelpRequest",
                "frPast8HelpRequest",
                "stlHintUsed",
                "past8BottomOut",
                "totalFrPercentPastWrong",
                "totalFrPastWrongCount",
                "frPast5WrongCount",
                "frPast8WrongCount",
                "totalFrTimeOnSkill",
                "timeSinceSkill",
                "frWorkingInSchool",
                "totalFrAttempted",
                "totalFrSkillOpportunities",
                "responseIsFillIn",
                "responseIsChosen",
                "endsWithScaffolding",
                "endsWithAutoScaffolding",
                "frTimeTakenOnScaffolding",
                "frTotalSkillOpportunitiesScaffolding",
                "totalFrSkillOpportunitiesByScaffolding",
                "frIsHelpRequestScaffolding",
                "timeGreater5Secprev2wrong",
                "sumRight",
                "helpAccessUnder2Sec",
                "timeGreater10SecAndNextActionRight",
                "consecutiveErrorsInRow",
                "sumTime3SDWhen3RowRight",
                "sumTimePerSkill",
                "totalTimeByPercentCorrectForskill",
                "Prev5count",
                "timeOver80",
                "manywrong",
                "confidence(BORED)",
                "confidence(CONCENTRATING)",
                "confidence(CONFUSED)",
                "confidence(FRUSTRATED)",
                "confidence(OFF TASK)",
                "confidence(GAMING)",
                "RES_BORED",
                "RES_CONCENTRATING",
                "RES_CONFUSED",
                "RES_FRUSTRATED",
                "RES_OFFTASK",
                "RES_GAMING",
                "Ln-1",
                "Ln",
                "MCAS",
                "Enrolled",
                "Selective",
                "isSTEM",
            ]
        )
        # 重命名列
        data = data.rename(
            columns={
                "studentId": "user_id",
                "problemId": "question_id",
                "correct": "label",
                "skill": "skill_id",
            }
        )
        # 将技能列映射为唯一的整数ID
        unique_skills = data["skill_id"].unique()
        skill_id_map = {skill: idx for idx, skill in enumerate(unique_skills)}
        data["skill_id"] = data["skill_id"].map(skill_id_map)

        # 按时间排序
        data = data.sort_values(by=["user_id", "startTime"])
        # 转换数据类型
        data["user_id"] = data["user_id"].astype(int)
        # 清除没有技能的问题
        data = data[data["skill_id"].notna()]
        # 清理label列中的异常值，只保留0和1
        data = data[data["label"].isin([0, 1])]
        # 重置索引
        data = data.reset_index(drop=True)
        # 限制序列长度
        data = DataSource.restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )
        # 将问题ID和技能ID转换为连续整数
        data = DataSource.map_to_continuous_ids(
            data, columns=["user_id", "question_id", "skill_id"]
        )
        self.processed_data = data

        # 保存元信息
        self.add_metadata("num_users", data["user_id"].nunique())
        self.add_metadata("num_questions", data["question_id"].nunique())
        self.add_metadata("num_skills", data["skill_id"].nunique())
        self.add_metadata("max_seq_len", self.args.max_seq_len)
        self.add_metadata("min_seq_len", self.args.min_seq_len)
        self.add_metadata("columns", data.columns.tolist())
