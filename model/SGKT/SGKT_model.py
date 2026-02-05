"""
SGKT model implementation.

Implements Session Graph-based Knowledge Tracing using:
- GCNConv for Heterogeneous Relation Graph (HRG)
- GatedGraphConv for Session Graph (SG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.core import MODELS
from ..layers import GeneralInteraction, HistoryRecap


class HRGEmbedding(nn.Module):
    """
    Heterogeneous Relation Graph (HRG) Embedding Module using GCNConv.

    The HRG graph contains:
    - Skill nodes and Question nodes
    - Edges: Question<->Skill (via skill_matrix) + Question<->Question (co-occurrence)

    Note: Outputs embedding_dim features (following original TF implementation).

    Args:
        num_skills: Number of skills
        num_questions: Number of questions
        embedding_dim: Node embedding dimension and output dimension
        num_layers: Number of GCN layers (corresponds to n-hop)
        max_neighbors: Maximum number of neighbors to extract per question
    """

    def __init__(
        self,
        num_skills,
        num_questions,
        embedding_dim,
        num_layers=2,
        max_neighbors=20,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_dim = embedding_dim
        self.max_neighbors = max_neighbors
        self.dropout = nn.Dropout(p=float(dropout))

        # Node embeddings
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim)
        self.question_embedding = nn.Embedding(num_questions, embedding_dim)

        # GCN layers - all layers maintain embedding_dim
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(embedding_dim, embedding_dim))

        # Precomputed neighbor indices (initialized in precompute_neighbors)
        self.register_buffer("neighbor_indices", None)  # [num_questions, max_neighbors]
        self.register_buffer("neighbor_mask", None)  # [num_questions, max_neighbors]
        self._precomputed = False

    def precompute_neighbors(self, edge_index):
        """
        Precompute neighbor indices for all questions.

        This method should be called once after model initialization,
        before training starts. It builds static neighbor index buffers
        that are reused across all forward passes.

        Args:
            edge_index: [2, E] HRG graph edge indices (static)
        """
        device = edge_index.device

        # 1. Extract question-question edges (vectorized)
        # Question nodes are indexed from num_skills to num_skills + num_questions - 1
        qq_mask = (edge_index[0] >= self.num_skills) & (
            edge_index[1] >= self.num_skills
        )
        qq_edges = edge_index[:, qq_mask].clone()
        qq_edges[0] -= self.num_skills
        qq_edges[1] -= self.num_skills

        # 2. Build adjacency list (dense format for efficiency)
        # For each question, store up to max_neighbors neighbor indices
        neighbor_indices = torch.full(
            (self.num_questions, self.max_neighbors),
            -1,
            dtype=torch.long,
            device=device,
        )
        neighbor_mask = torch.zeros(
            (self.num_questions, self.max_neighbors),
            dtype=torch.bool,
            device=device,
        )

        # 3. Extract neighbors for each question with deterministic order
        # Match the original implementation's ordering: src matches first, then dst matches
        # NOTE: Original implementation has a bug where it collects src[src==q] instead of
        # dst[src==q], but we maintain this behavior for numerical equivalence
        if qq_edges.shape[1] > 0:
            src = qq_edges[0]
            dst = qq_edges[1]

            # Build neighbor lists matching original implementation's behavior
            for q_id in range(self.num_questions):
                # Original logic: src[src == q] gives the source nodes (including q itself)
                # Then dst[dst == q] gives the destination nodes
                # This is intentionally kept as-is to match original behavior
                src_matches = src[src == q_id]
                dst_matches = dst[dst == q_id]
                # Concatenate: src matches first, then dst matches
                neighbors = torch.cat([src_matches, dst_matches])
                # Remove self-loops only (no deduplication to match original)
                neighbors = neighbors[neighbors != q_id]
                # Limit to max_neighbors
                if len(neighbors) > self.max_neighbors:
                    neighbors = neighbors[: self.max_neighbors]

                n = len(neighbors)
                if n > 0:
                    neighbor_indices[q_id, :n] = neighbors
                    neighbor_mask[q_id, :n] = True

        # 4. Register as buffers
        self.register_buffer("neighbor_indices", neighbor_indices)
        self.register_buffer("neighbor_mask", neighbor_mask)
        self._precomputed = True

    def forward(self, hrg_data, question_indices):
        """
        Forward pass for HRG embedding.

        Args:
            hrg_data: PyG Data object with edge_index
            question_indices: [batch_size, seq_len] Question indices in batch

        Returns:
            question_features: [batch_size, seq_len, embedding_dim]
            neighbor_features: [batch_size, seq_len, max_neighbors, embedding_dim]
        """
        device = question_indices.device
        batch_size, seq_len = question_indices.shape

        # Auto-precompute neighbors if not done yet
        if not self._precomputed:
            self.precompute_neighbors(hrg_data.edge_index)

        # 1. Build initial node features
        num_nodes = self.num_skills + self.num_questions
        x = torch.zeros(num_nodes, self.embedding_dim, device=device)

        # Skill node features: [0, num_skills)
        x[: self.num_skills] = self.skill_embedding.weight

        # Question node features: [num_skills, num_skills + num_questions)
        x[self.num_skills :] = self.question_embedding.weight

        # 2. Multi-layer GCN
        edge_index = hrg_data.edge_index.to(device)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # 3. Extract question features for current batch
        # Convert question indices to global node IDs
        question_node_indices = question_indices + self.num_skills  # [B, S]

        # Gather features
        question_features = x[question_node_indices]  # [B, S, embedding_dim]

        # 4. Extract neighbor features for next neighbor sampling
        neighbor_features = self._extract_neighbor_features(
            x, edge_index, question_node_indices, device
        )  # [B, S, max_neighbors, embedding_dim]

        return question_features, neighbor_features

    def _extract_neighbor_features(self, x, edge_index, question_node_indices, device):
        """
        Extract neighbor features for each question using precomputed indices.

        This optimized version uses precomputed neighbor indices to avoid
        building adjacency lists during each forward pass.

        Args:
            x: [num_nodes, embedding_dim] All node features after GCN
            edge_index: [2, num_edges] Edge indices (kept for API compatibility)
            question_node_indices: [batch_size, seq_len] Question node IDs
            device: Torch device

        Returns:
            neighbor_features: [batch_size, seq_len, max_neighbors, embedding_dim]
        """
        batch_size, seq_len = question_node_indices.shape

        # Convert question node indices to question indices (0-based)
        q_indices = question_node_indices - self.num_skills  # [B, S]

        # Gather precomputed neighbor indices [B, S, K]
        batch_neighbor_idx = self.neighbor_indices[q_indices]
        batch_neighbor_mask = self.neighbor_mask[q_indices]

        # Convert to global node IDs [B, S, K]
        batch_neighbor_idx = batch_neighbor_idx + self.num_skills
        # Set padding positions to 0 (valid index, will be masked out later)
        batch_neighbor_idx = torch.where(
            batch_neighbor_mask,
            batch_neighbor_idx,
            torch.zeros_like(batch_neighbor_idx),
        )

        # Vectorized gather: x[neighbor_idx] -> [B, S, K, E]
        # Use advanced indexing
        neighbor_features = x[batch_neighbor_idx]  # [B, S, K, E]

        # Apply mask to zero out padding positions
        neighbor_features = (
            neighbor_features * batch_neighbor_mask.unsqueeze(-1).float()
        )

        return neighbor_features


class SGEmbedding(nn.Module):
    """
    Session Graph (SG) Embedding Module using GRUCell.

    Following the original TF implementation which uses GRUCell for sequential
    processing. This ensures unidirectional information flow where timestep t
    only depends on previous timesteps.

    Note: The original TF implementation assumes embedding_dim == hidden_dim.
    If they differ, we add projection layers to handle the mismatch.

    Args:
        embedding_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension (for input_trans_embedding)
    """

    def __init__(self, embedding_dim, hidden_dim, num_layers=2, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=float(dropout))

        # GRUCell - following TF implementation (line 136)
        # Output dimension is embedding_dim, not hidden_dim
        self.gru_cell = nn.GRUCell(
            input_size=2 * embedding_dim,  # av dimension
            hidden_size=embedding_dim,
        )

        # Gate parameters (compatible with TF implementation)
        self.W_in = nn.Linear(embedding_dim, embedding_dim)
        self.W_out = nn.Linear(embedding_dim, embedding_dim)
        self.b_in = nn.Parameter(torch.zeros(embedding_dim))
        self.b_out = nn.Parameter(torch.zeros(embedding_dim))

        # Projection layer for hidden_dim -> embedding_dim if needed
        # Original TF assumes E == H, but we handle E != H gracefully
        if embedding_dim != hidden_dim:
            self.hidden_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        else:
            self.hidden_to_embedding = nn.Identity()

    def forward(
        self,
        question_emb,
        answer_emb,
        input_trans_embedding,
        user_sequence,
        user_mask,
    ):
        """
        Forward pass for session graph embedding using GRUCell.

        Args:
            question_emb: [batch_size, seq_len, embedding_dim]
            answer_emb: [batch_size, seq_len, embedding_dim]
            input_trans_embedding: [batch_size, seq_len, hidden_dim] Features from input transformation
            user_sequence: [batch_size, seq_len] Question IDs
            user_mask: [batch_size, seq_len] Valid position mask

        Returns:
            output_series: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = question_emb.shape
        device = question_emb.device

        # Pre-compute valid lengths for all batches
        valid_lens = user_mask.sum(dim=1).long()  # [B]
        max_valid_len = valid_lens.max().item()

        # Handle case where all sequences are empty
        if max_valid_len == 0:
            return torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)

        # Initialize hidden states: fin_state = answer_emb[:, 0] for each batch
        fin_states = answer_emb[:, 0, :]  # [B, E]

        # Prepare output tensor
        outputs = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)

        # Process each timestep
        for t in range(max_valid_len):
            # Create mask for batches that are still valid at this timestep
            valid_mask = t < valid_lens  # [B]

            if not valid_mask.any():
                break

            # Get features for current timestep (all batches)
            q_feat = question_emb[:, t, :]  # [B, E]
            a_feat = answer_emb[:, t, :]  # [B, E]
            it_feat = input_trans_embedding[:, t, :]  # [B, H]
            it_feat_projected = self.hidden_to_embedding(it_feat)  # [B, E]

            # Compute in_state
            in_state = q_feat + a_feat + it_feat_projected  # [B, E]
            in_state = self.dropout(in_state)

            # Compute gates
            fin_state_in = self.W_in(in_state) + self.b_in  # [B, E]
            fin_state_out = self.W_out(in_state) + self.b_out  # [B, E]

            # Compute av
            av = torch.cat([fin_state_in, fin_state_out], dim=-1)  # [B, 2E]
            av = av + torch.cat([in_state, in_state], dim=-1)  # [B, 2E]

            # Apply GRUCell to all batches simultaneously
            new_fin_states = self.gru_cell(av, fin_states)  # [B, E]
            new_fin_states = self.dropout(new_fin_states)

            # Update only valid positions
            fin_states = torch.where(
                valid_mask.unsqueeze(-1).expand_as(fin_states),
                new_fin_states,
                fin_states,
            )

            # Store output for this timestep
            outputs[:, t, :] = fin_states

        return outputs


class NextNeighborSampler(nn.Module):
    """
    Next Neighbor Sampler from aggregated HRG embeddings.

    Samples N neighbors from the given features following the TensorFlow implementation:
    - If enough neighbors: randomly shuffle once and take first N
    - If not enough neighbors: tile (repeat) the neighbors to reach N

    This matches the original TF implementation (model.py:280-290).

    Args:
        next_neighbor_num: Number of neighbors to sample (N)
    """

    def __init__(self, next_neighbor_num):
        super().__init__()
        self.next_neighbor_num = next_neighbor_num

    def forward(self, neighbor_features):
        """
        Sample neighbors from pre-aggregated HRG neighbor features.

        Args:
            neighbor_features: [batch_size, seq_len, total_neighbors, emb_dim]
                              Pre-aggregated neighbor features from HRG graph

        Returns:
            next_neighbors: [batch_size, seq_len, next_neighbor_num, emb_dim]
        """
        batch_size, seq_len, total_neighbors, emb_dim = neighbor_features.shape

        if total_neighbors == 0:
            # No neighbors available, return zeros
            return torch.zeros(
                batch_size,
                seq_len,
                self.next_neighbor_num,
                emb_dim,
                device=neighbor_features.device,
                dtype=neighbor_features.dtype,
            )

        perm = torch.randperm(total_neighbors, device=neighbor_features.device)
        if total_neighbors >= self.next_neighbor_num:
            # Enough neighbors: directly sample first N using advanced indexing
            sampled_idx = perm[: self.next_neighbor_num]
            next_neighbors = neighbor_features[:, :, sampled_idx, :]
        else:
            # Not enough neighbors: need to tile (repeat) indices to reach N
            repeat_times = -(
                -self.next_neighbor_num // total_neighbors
            )  # Ceiling division
            extended_idx = perm.repeat(repeat_times)[: self.next_neighbor_num]
            next_neighbors = neighbor_features[:, :, extended_idx, :]
        return next_neighbors


class SelfAttentionHistory(nn.Module):
    """
    Self-Attention History module.

    This implements the hist_neighbor_sampler function from model.py:218-248.

    Args:
        hidden_dim: Hidden state dimension (H)
        seq_len: Maximum sequence length (S)
        hist_neighbor_num: Number of historical neighbors to sample (M)
    """

    def __init__(self, hidden_dim, seq_len, hist_neighbor_num, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=float(dropout))
        self.seq_len = seq_len
        self.hist_neighbor_num = hist_neighbor_num

        # Feature-wise interaction parameters (from TF: xita, xt1, xt2)
        # xita: [seq_len] - position-wise bias
        self.register_parameter("xita", nn.Parameter(torch.empty(seq_len)))
        # xt1: [seq_len, seq_len] - first transformation matrix
        self.register_parameter("xt1", nn.Parameter(torch.empty(seq_len, seq_len)))
        # xt2: [seq_len, seq_len] - second transformation matrix
        self.register_parameter("xt2", nn.Parameter(torch.empty(seq_len, seq_len)))

        # Q, K, V projection matrices (from TF: K, Q, V)
        # These are Linear layers without bias (bias is added separately)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Bias term for QKV projections: [seq_len + 1, hidden_dim]
        # +1 because we add zero padding at the beginning
        self.register_parameter(
            "bias_kv", nn.Parameter(torch.empty(seq_len + 1, hidden_dim))
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following TF implementation (uniform initialization)."""
        stdv = 1.0 / (self.hidden_dim**0.5)
        nn.init.uniform_(self.xita, -stdv, stdv)
        nn.init.uniform_(self.xt1, -stdv, stdv)
        nn.init.uniform_(self.xt2, -stdv, stdv)
        nn.init.uniform_(self.bias_kv, -stdv, stdv)

        # Initialize Q, K, V with Xavier
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, input_embedding, hist_neighbor_index):
        """
        Forward pass implementing TF's hist_neighbor_sampler logic.

        Args:
            input_embedding: [batch_size, seq_len, hidden_dim] Input states
            hist_neighbor_index: [batch_size, seq_len, hist_neighbor_num]
                                Pre-computed historical neighbor indices
                                Following TF implementation, these indices
                                are sampled based on skill matching

        Returns:
            hist_neighbors_features: [batch_size, seq_len, hist_neighbor_num, hidden_dim]
                                    Attention-enhanced historical features
        """
        batch_size, seq_len, hidden_dim = input_embedding.shape
        device = input_embedding.device
        dtype = input_embedding.dtype

        input_embedding = self.dropout(input_embedding)

        # For each dimension i: result[:, :, i] = exp((x[:, :, i] - x[:, :, 0]) @ xt1) @ xt2 + xita

        # Step 1: Compute diff for all dimensions: [B, S, H] - [B, S, 1] -> [B, S, H]
        diff = input_embedding - input_embedding[:, :, 0:1]

        # Step 2: First matrix multiplication for all dimensions
        # [B, S, H] @ [S, S] -> [B, S, H] using einsum: bsh,st->bth
        transformed1 = torch.einsum("bsh,st->bth", diff, self.xt1)
        exp_transformed = torch.exp(transformed1)

        # Step 3: Second matrix multiplication for all dimensions
        transformed2 = torch.einsum("bsh,st->bth", exp_transformed, self.xt2)

        # Step 4: Add position-wise bias (broadcast xita: [S] -> [B, S, H])
        input_embedding_transformed = transformed2 + self.xita.unsqueeze(0).unsqueeze(
            -1
        )
        input_embedding_transformed = self.dropout(input_embedding_transformed)

        # Add zero padding
        zero_padding = torch.zeros(
            batch_size, 1, hidden_dim, device=device, dtype=dtype
        )
        input_emb_padded = torch.cat(
            [input_embedding_transformed, zero_padding], dim=1
        )  # [B, S+1, H]

        # QKV Self-Attention
        bias_expanded = self.bias_kv.unsqueeze(0)  # [1, S+1, H]
        EK = self.K(input_emb_padded) + bias_expanded  # [B, S+1, H]
        EQ = self.Q(input_emb_padded) + bias_expanded  # [B, S+1, H]
        EV = self.V(input_emb_padded) + bias_expanded  # [B, S+1, H]

        # Compute attention scores
        A = torch.bmm(EQ, EK.transpose(1, 2)) / torch.sqrt(
            torch.tensor(hidden_dim, dtype=torch.float32, device=device)
        )  # [B, S+1, S+1]

        seq_len_plus_1 = seq_len + 1
        causal_mask = torch.tril(
            torch.ones(seq_len_plus_1, seq_len_plus_1, device=device, dtype=torch.bool)
        )
        A = A.masked_fill(~causal_mask, 0.0)

        # Apply attention to values
        B = torch.bmm(A, EV)  # [B, S+1, H]
        B = self.dropout(B)

        # Gather using hist_neighbor_index
        temp_hist_index = hist_neighbor_index.reshape(
            batch_size, seq_len * self.hist_neighbor_num
        )
        temp_hist_index_clamped = torch.clamp(temp_hist_index, 0, seq_len)  # [B, S*M]

        batch_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, seq_len * self.hist_neighbor_num)
        )  # [B, S*M]

        hist_neighbors_features_flat = B[
            batch_indices, temp_hist_index_clamped, :
        ]  # [B, S*M, H]

        hist_neighbors_features = hist_neighbors_features_flat.reshape(
            batch_size, seq_len, self.hist_neighbor_num, hidden_dim
        )

        return hist_neighbors_features


@MODELS.register("SGKT")
class SGKT(nn.Module):
    """
    Session Graph-based Knowledge Tracing (SGKT) Model.

    Dual graph architecture:
    1. HRG (Heterogeneous Relation Graph): Static question-skill graph with GCNConv
    2. SG (Session Graph): Dynamic session graph with GatedGraphConv

    Args:
        args: Model arguments
        data_metadata: Data source metadata
    """

    def __init__(self, args, data_metadata, **kwargs):
        super().__init__(**kwargs)

        self.num_skills = data_metadata["num_skills"]
        self.num_questions = data_metadata["num_questions"]
        self.max_seq_len = data_metadata["max_seq_len"]
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.dropout_p = args.dropout
        self.dropout_gnn_p = args.dropout_gnn
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.dropout_gnn = nn.Dropout(p=self.dropout_gnn_p)

        # Embeddings
        self.question_embedding = nn.Embedding(self.num_questions, self.embedding_dim)
        self.skill_embedding = nn.Embedding(self.num_skills, self.embedding_dim)
        self.answer_embedding = nn.Embedding(2, self.embedding_dim)

        # HRG Embedding Module (GCNConv)
        self.hrg_embedding = HRGEmbedding(
            num_skills=self.num_skills,
            num_questions=self.num_questions,
            embedding_dim=self.embedding_dim,
            num_layers=getattr(args, "n_hop", 2),
            max_neighbors=50,
            dropout=self.dropout_gnn_p,
        )

        # Feature transformation layers
        self.feature_trans = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.feature_trans_activation = nn.ReLU()

        # Input transformation: combines HRG features with answer embeddings
        self.input_trans = nn.Linear(
            self.hidden_dim + self.embedding_dim, self.hidden_dim
        )

        # SG Embedding Module (GRUCell)
        self.sg_embedding = SGEmbedding(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=getattr(args, "sg_layers", 2),
            dropout=self.dropout_p,
        )

        self.hist_sampler = HistoryRecap(
            hist_neighbor_num=getattr(args, "hist_neighbor_num", 5),
            att_bound=getattr(args, "att_bound", 0.7),
        )

        # Self-Attention History Module (E_answring_states)
        self.self_attention = SelfAttentionHistory(
            hidden_dim=self.hidden_dim,
            seq_len=self.max_seq_len,
            hist_neighbor_num=getattr(args, "hist_neighbor_num", 5),
            dropout=self.dropout_p,
        )

        self.next_sampler = NextNeighborSampler(
            next_neighbor_num=getattr(args, "next_neighbor_num", 5)
        )

        self.general_interaction = GeneralInteraction(self.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        user_sequence,
        user_response,
        user_mask,
        hrg_data,
        hist_neighbor_index=None,
    ):
        """
        Forward pass of SGKT model.

        Args:
            user_sequence: [batch_size, seq_len] Question IDs
            user_response: [batch_size, seq_len] Responses (0/1)
            user_mask: [batch_size, seq_len] Valid position mask
            hrg_data: PyG Data object for HRG graph
            hist_neighbor_index: [batch_size, seq_len, M] Pre-computed fallback indices
                                 Following TF implementation (model.py:274)

        Returns:
            logits: [batch_size, seq_len-1] Prediction logits
        """
        # 1. Basic embeddings
        question_embs = self.dropout(
            self.question_embedding(user_sequence)
        )  # [B, S, E]
        answer_embs = self.dropout(self.answer_embedding(user_response))  # [B, S, E]

        # 2. HRG Embedding
        hrg_features, hrg_neighbor_features = self.hrg_embedding(
            hrg_data, user_sequence
        )  # [B, S, E], [B, S, max_neighbors, E]
        hrg_features = self.dropout_gnn(hrg_features)
        hrg_neighbor_features = self.dropout_gnn(hrg_neighbor_features)

        # 3. Feature transformation
        feature_trans_embedding = self.feature_trans_activation(
            self.feature_trans(hrg_features)
        )  # [B, S, H]
        feature_trans_embedding = self.dropout_gnn(feature_trans_embedding)

        # 4. Input transformation
        input_fa_embedding = torch.cat(
            [feature_trans_embedding, answer_embs], dim=-1
        )  # [B, S, H+E]
        input_trans_embedding = self.input_trans(input_fa_embedding)  # [B, S, H]
        input_trans_embedding = self.dropout(input_trans_embedding)

        # 5. Next feature transformation
        next_hrg_features = hrg_features[:, 1:, :]  # [B, S-1, E]
        next_feature_trans_embedding = self.feature_trans_activation(
            self.feature_trans(next_hrg_features)
        )  # [B, S-1, H]
        next_feature_trans_embedding = self.dropout_gnn(next_feature_trans_embedding)

        # 6. SG Embedding
        sg_features = self.sg_embedding(
            question_embs,
            answer_embs,
            input_trans_embedding,
            user_sequence,
            user_mask,
        )  # [B, S, E]
        sg_features = self.dropout(sg_features)

        # 7. Self-Attention History
        E_answring_states = self.self_attention(
            input_embedding=input_trans_embedding,
            hist_neighbor_index=hist_neighbor_index,
        )  # [B, S, M, H]

        # 8. Historical Neighbor Sampling
        hist_neighbor_index_shifted = None
        if hist_neighbor_index is not None:
            hist_neighbor_index_shifted = hist_neighbor_index[:, :-1, :]

        input_trans_embedding_shifted = input_trans_embedding[:, :-1, :]  # [B, S-1, H]

        hist_neighbors = self.hist_sampler(
            input_q_emb=question_embs[:, :-1, :],  # [B, S-1, E]
            next_q_emb=question_embs[:, 1:, :],  # [B, S-1, E]
            qa_emb=input_trans_embedding_shifted,  # [B, S-1, H]
            user_mask=user_mask[:, :-1],  # [B, S-1]
            hist_neighbor_index=hist_neighbor_index_shifted,  # [B, S-1, M]
        )  # [B, S-1, M, H]
        hist_neighbors = self.dropout(hist_neighbors)

        # 9. Combine hist_neighbors with E_answring_states
        hist_neighbors_combined = (
            hist_neighbors + E_answring_states[:, :-1, :, :]
        )  # [B, S-1, M, H]
        hist_neighbors_combined = self.dropout(hist_neighbors_combined)

        # 10. Next Neighbor Sampling
        next_hrg_neighbor_features = hrg_neighbor_features[
            :, 1:, :, :
        ]  # [B, S-1, max_neighbors, E]

        next_hrg_neighbor_features_trans = self.feature_trans_activation(
            self.feature_trans(next_hrg_neighbor_features)
        )  # [B, S-1, max_neighbors, H]
        next_hrg_neighbor_features_trans = self.dropout_gnn(
            next_hrg_neighbor_features_trans
        )

        next_neighbors = self.next_sampler(
            next_hrg_neighbor_features_trans
        )  # [B, S-1, N, H]
        next_neighbors = self.dropout_gnn(next_neighbors)

        # 11. Prediction Module (FM + Attention)
        if self.embedding_dim != self.hidden_dim:
            sg_features_projected = self.feature_trans_activation(
                self.feature_trans(sg_features[:, :-1])
            )  # [B, S-1, H]
        else:
            sg_features_projected = sg_features[:, :-1]  # [B, S-1, E=H]
        sg_features_projected = self.dropout(sg_features_projected)

        student_status = torch.cat(
            [
                sg_features_projected.unsqueeze(2),  # [B, S-1, 1, H]
                hist_neighbors_combined,  # [B, S-1, M, H]
            ],
            dim=2,
        )  # [B, S-1, M+1, H]
        student_status = self.dropout(student_status)

        knowledge_status = torch.cat(
            [
                next_feature_trans_embedding.unsqueeze(2),  # [B, S-1, 1, H]
                next_neighbors,  # [B, S-1, N, H]
            ],
            dim=2,
        )  # [B, S-1, N+1, H]
        knowledge_status = self.dropout_gnn(knowledge_status)

        logits = self.general_interaction(
            student_status,
            knowledge_status,
            user_mask[:, :-1],
        )  # [B, S-1]

        return logits
