"""extraKTQ: question-level extraKT data for the skill-vs-question ablation."""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class extraKTQDataset(Dataset):
    """Question-level extraKT sample: ``(question, response, mask)``."""

    def __init__(self, questions, responses, masks):
        self.questions = torch.from_numpy(questions).long()
        self.responses = torch.from_numpy(responses).long()
        self.masks = torch.from_numpy(masks).bool()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        return self.questions[idx], self.responses[idx], self.masks[idx]


class extraKTQModelData(QuestionModelData):
    """Question-level data for the extraKT ablation.

    The question id is the concept embedding unit (no Rasch pid), to contrast
    with the skill-level extraKT. Test uses the standard K-fold split instead
    of windowlate, which is specific to KC (skill-level) models.
    """

    def __init__(self, data_src):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        user_question, user_response, user_mask, _ = self.load_sequence_data()

        train_data, val_data, test_data = self.split_kfold_data(
            user_question, user_response, user_mask, fold_idx=rc.data.fold
        )

        train_dataset = extraKTQDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = extraKTQDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = extraKTQDataset(test_data[0], test_data[1], test_data[2])

        logger.info(
            f"extraKTQ data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset
