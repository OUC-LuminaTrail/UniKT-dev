"""Precomputed per-question related-skill id table for HDHKT.

Replaces the per-batch ``question_skill_matrix[next_user_sequence]`` dense
gather + ``argsort`` used to extract each question's related skill indices.
The table is built once from the binary question-skill matrix in data
preparation and indexed in the forward pass.
"""

import torch


def build_skill_index_table(
    question_skill_matrix: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Build a padded table of related-skill ids per question.

    Args:
        question_skill_matrix: Binary matrix of shape ``[num_questions, num_skills]``;
            a value of 1 means the skill is related to the question.
        padding_index: Sentinel id written to unused slots. Must be outside
            ``[0, num_skills)`` (typically ``num_skills``); the forward pass
            appends a zero embedding at this row index.

    Returns:
        ``skill_ids_per_question`` of shape ``[num_questions, K_max]`` (``long``),
        where ``K_max`` is the maximum number of related skills over all
        questions. Row ``q`` holds the related skill ids in ascending order in
        columns ``[0, count[q])`` and ``padding_index`` in the remaining columns.
        If every question has zero related skills, the second dimension is 0.
    """
    if question_skill_matrix.dim() != 2:
        raise ValueError(
            f"question_skill_matrix must be 2-D, got shape {tuple(question_skill_matrix.shape)}"
        )

    num_questions, num_skills = question_skill_matrix.shape
    mask = question_skill_matrix > 0
    counts = mask.sum(dim=1).long()  # [Q]
    k_max = int(counts.max().item()) if num_questions > 0 else 0

    skill_ids = torch.full(
        (num_questions, k_max),
        padding_index,
        dtype=torch.long,
        device=question_skill_matrix.device,
    )

    if k_max == 0:
        return skill_ids

    # torch.nonzero yields (row, col) pairs in row-major (ascending) order, so
    # within each question row the skill ids are already ascending. Assign each
    # pair to column = its rank within the row.
    nz = torch.nonzero(mask, as_tuple=False)  # [P, 2]
    if nz.numel() == 0:
        return skill_ids

    rows = nz[:, 0]
    cols = nz[:, 1]
    offsets = torch.zeros(num_questions + 1, dtype=torch.long, device=counts.device)
    offsets[1:] = counts.cumsum(0)
    rank = (
        torch.arange(nz.size(0), dtype=torch.long, device=counts.device) - offsets[rows]
    )
    skill_ids[rows, rank] = cols
    return skill_ids
