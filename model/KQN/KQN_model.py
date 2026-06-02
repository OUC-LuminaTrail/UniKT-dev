"""KQN baseline model.

Knowledge Query Network encodes past concept-response interactions into a
knowledge-state vector and queries that state with the next concept vector.
"""

import torch
from torch import nn
from torch.nn import functional as F


class KQN(nn.Module):
    """Knowledge Query Network.

    Args:
        num_skills: Number of 0-based KC/concept ids.
        n_hidden: Shared dimensionality of knowledge-state and skill vectors.
        n_rnn_hidden: Hidden size of the recurrent knowledge encoder.
        n_mlp_hidden: Hidden size of the skill encoder MLP.
        n_rnn_layers: Number of recurrent layers.
        rnn_type: Recurrent cell type, either ``"lstm"`` or ``"gru"``.
        dropout: Dropout applied to encoded knowledge states before querying.

    Tensor semantics:
        concept, response, next_concept: int tensors with shape [B, L].
        The output has shape [B, L], where output[:, t] estimates
        P(response at next_concept[:, t] is correct | concept/response history
        through t).
    """

    def __init__(
        self,
        num_skills: int,
        n_hidden: int,
        n_rnn_hidden: int,
        n_mlp_hidden: int,
        n_rnn_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_skills = num_skills
        self.n_hidden = n_hidden
        self.n_rnn_hidden = n_rnn_hidden
        self.n_mlp_hidden = n_mlp_hidden
        self.n_rnn_layers = n_rnn_layers
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == "lstm":
            rnn_cls = nn.LSTM
        elif self.rnn_type == "gru":
            rnn_cls = nn.GRU
        else:
            raise ValueError("rnn_type must be either 'lstm' or 'gru'")

        self.rnn = rnn_cls(
            input_size=2 * num_skills,
            hidden_size=n_rnn_hidden,
            num_layers=n_rnn_layers,
            batch_first=True,
        )
        self.knowledge_projection = nn.Linear(n_rnn_hidden, n_hidden)

        # PyKT-style trainable skill encoder over one-hot KC ids. The one-hot
        # tensor is created from concept ids only; no external skill features.
        self.skill_encoder = nn.Sequential(
            nn.Linear(num_skills, n_mlp_hidden),
            nn.ReLU(),
            nn.Linear(n_mlp_hidden, n_hidden),
            nn.ReLU(),
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        concept: torch.Tensor,
        response: torch.Tensor,
        next_concept: torch.Tensor,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run KQN on shifted concept-response sequences.

        Args:
            concept: Current concept ids, shape [B, L].
            response: Current binary responses, shape [B, L].
            next_concept: Next concept ids to query, shape [B, L].
            return_states: If true, also return encoded knowledge and skills.

        Returns:
            Predicted probabilities with shape [B, L]. If ``return_states`` is
            true, returns ``(probabilities, encoded_knowledge, encoded_skills)``.
        """
        interaction = self.encode_interactions(concept, response)
        encoded_knowledge = self.encode_knowledge(interaction)
        encoded_skills = self.encode_skills(next_concept)
        encoded_knowledge = self.dropout_layer(encoded_knowledge)

        logits = torch.sum(encoded_knowledge * encoded_skills, dim=-1)
        probs = torch.sigmoid(logits)
        if return_states:
            return probs, encoded_knowledge, encoded_skills
        return probs

    def encode_interactions(
        self, concept: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        """Encode (concept, response) as PyKT-style 2N one-hot interactions."""
        interaction_id = response.long() * self.num_skills + concept.long()
        return F.one_hot(interaction_id, num_classes=2 * self.num_skills).float()

    def encode_knowledge(self, interaction: torch.Tensor) -> torch.Tensor:
        """Encode interaction history into knowledge-state vectors."""
        rnn_output, _ = self.rnn(interaction)
        return self.knowledge_projection(rnn_output)

    def encode_skills(self, next_concept: torch.Tensor) -> torch.Tensor:
        """Encode next-concept ids into positive, L2-normalized skill vectors."""
        next_skills = F.one_hot(
            next_concept.long(), num_classes=self.num_skills
        ).float()
        skill_vectors = self.skill_encoder(next_skills)
        return F.normalize(skill_vectors, p=2, dim=-1)
