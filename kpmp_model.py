import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

class PositionalEncoding(nn.Module):
    """Encode tile coordinates"""
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, coords):
        return self.mlp(coords)


class CrossAttentionBlock(nn.Module):
    """Cross-attention between glom-tiles and lesions"""
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.head_dim = dim_q // num_heads
        
        assert self.dim_q % self.num_heads == 0, "dim_q must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, return_attention=False):
        """
        Args:
            query: [B, G, dim_q] - glom-tile embeddings
            key: [B, L, dim_kv] - lesion embeddings
            value: [B, L, dim_kv] - lesion embeddings
            return_attention: whether to return raw attention weights
        Returns:
            output: [B, G, dim_q]
            attention_weights: [B, G, L] if return_attention=True
        """
        B, G, _ = query.shape
        L = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(B, G, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, G, self.dim_q)
        out = self.out_proj(out)
        
        if return_attention:
            # Average attention weights across heads
            attn_weights_avg = attn_weights.mean(dim=1)  # [B, G, L]
            return out, attn_weights_avg
        
        return out, None


class ConceptTransformer(nn.Module):
    """
    Concept Transformer
    
    Key components:
    - X_{G-T}: Glom-tile embeddings from first cross-attention
    - X_L: Fixed lesion embeddings (pathologist-defined descriptors)
    - A: Attention matrix from CT block
    - V: Value matrix (lesion embeddings)
    - O^(t): Task-specific projection matrices
    - γ: Lesion relevance scores (patient-specific)
    - β: Lesion importance weights (task/class-specific)
    """
    def __init__(
        self,
        glom_dim: int = 768,
        tile_dim: int = 1536,
        lesion_embed_dim: int = 768,
        num_lesions: int = 20,  # L: number of lesion
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.glom_dim = glom_dim
        self.tile_dim = tile_dim
        self.lesion_embed_dim = lesion_embed_dim
        self.num_lesions = num_lesions
        
        assert glom_dim == lesion_embed_dim, f"Dimensions must match: glom_dim={glom_dim}, lesion_embed_dim={lesion_embed_dim}"
        self.d_m = glom_dim  
        
        # Positional encoding for tiles
        self.pos_encoder = PositionalEncoding(
            input_dim=2,
            hidden_dim=256,
            output_dim=tile_dim
        )
        
        # Fusion layer for tiles + positional encoding
        self.tile_fusion = nn.Sequential(
            nn.Linear(tile_dim * 2, tile_dim),
            nn.LayerNorm(tile_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # First Cross-attention: Glom to WSI tiles
        self.cross_attn_glom_tile = CrossAttentionBlock(
            dim_q=glom_dim,
            dim_kv=tile_dim,
            num_heads=num_heads
        )
        
        # Feed-forward after glom-tile attention
        self.ffn_glom_tile = self._build_ffn(glom_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(glom_dim)
        self.norm2 = nn.LayerNorm(glom_dim)
        
        # Core CT Block: Cross-attention from Glom-tiles to Lesions
        # Query:(glom-tile embeddings)
        # Key & Value: (lesion embeddings)
        self.cross_attn_ct = CrossAttentionBlock(
            dim_q=self.d_m,
            dim_kv=self.d_m,
            num_heads=num_heads
        )
        
        # Task-specific output matrices O^(t)
        
        # 1. DM-gate: Binary (n_c = 1)
        self.O_dm_gate = nn.Parameter(torch.randn(self.d_m, 1))
        
        # 2. DM-subtype: 3 classes (n_c = 3)
        self.O_dm_sub = nn.Parameter(torch.randn(self.d_m, 3))
        
        # 3. No-DM-subtype: 2 classes (n_c = 2)
        self.O_no_dm_sub = nn.Parameter(torch.randn(self.d_m, 2))
        
        # 4. Outcome: Binary (n_c = 1)
        self.O_outcome = nn.Parameter(torch.randn(self.d_m, 1))
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize output matrices with small values"""
        nn.init.xavier_uniform_(self.O_dm_gate, gain=0.1)
        nn.init.xavier_uniform_(self.O_dm_sub, gain=0.1)
        nn.init.xavier_uniform_(self.O_no_dm_sub, gain=0.1)
        nn.init.xavier_uniform_(self.O_outcome, gain=0.1)
    
    def _build_ffn(self, dim, hidden_dim, dropout):
        """Build feed-forward network"""
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def compute_gamma(self, attention_weights, glom_mask=None):
        """
        Compute lesion relevance scores γ_l(x)
        
        Args:
            attention_weights: [B, G, L] - attention matrix A from CT block
            glom_mask: [B, G] - mask for valid glomeruli (True = masked)
        
        Returns:
            gamma: [B, L] - patient-specific lesion relevance scores
        """
        B, G, L = attention_weights.shape
        
        if glom_mask is not None:
            # Apply mask: set masked positions to 0
            valid_mask = (~glom_mask).float().unsqueeze(-1)  # [B, G, 1]
            masked_attention = attention_weights * valid_mask
            
            # Sum and normalize by number of valid gloms
            gamma = masked_attention.sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        else:
            # Simple average across all gloms
            gamma = attention_weights.mean(dim=1)
        
        return gamma
    
    def compute_logits(self, gamma, V, O_matrix):
        """
        Compute task-specific logits using concept bottleneck
        logit_i^(t) = Σ_l β_{l,i}^(t) * γ_l(x)
        where β_{l,i}^(t) = [V·O^(t)]_{l,i}
        
        Args:
            gamma: [B, L] - lesion relevance scores (patient-specific)
            V: [B, L, d_m] - lesion embeddings (Value matrix from cross-attention)
            O_matrix: [d_m, n_c] - task-specific projection matrix
        
        Returns:
            logits: [B, n_c] - class logits for task t
            beta: [B, L, n_c] - lesion importance weights for interpretability
        """
        # Compute β = V @ O^(t)
        # This tells us how each lesion contributes to each class
        beta = torch.matmul(V, O_matrix)  # [B, L, n_c]
        
        # Expand gamma for broadcasting
        gamma_expanded = gamma.unsqueeze(-1)  # [B, L, 1]
        
        # Compute logits = Σ_l β_{l,i} * γ_l
        logits = (beta * gamma_expanded).sum(dim=1)  # [B, n_c]
        
        return logits, beta
    
    def forward(
        self,
        glom_embs,
        tile_embs,
        coords,
        lesion_embeddings,
        glom_mask=None,
        tile_mask=None,
        return_attention=True
    ):
        """
        Forward pass through Concept Transformer
        
        Args:
            glom_embs: [B, G, glom_dim] - Original glomeruli embeddings
            tile_embs: [B, T, tile_dim] - WSI tile embeddings
            coords: [B, T, 2] - Tile coordinates
            lesion_embeddings: [B, L, lesion_embed_dim] - Fixed lesion embeddings
                These are the SAME L lesion prototypes for all gloms/patients
            glom_mask: [B, G] - Mask for valid glomeruli (True = masked)
            tile_mask: [B, T] - Mask for valid tiles (True = masked)
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with predictions and intermediate outputs
        """
        B, G, _ = glom_embs.shape
        L = lesion_embeddings.shape[1]
        
        # Stage 1: Process tiles with positional encoding
        pos_emb = self.pos_encoder(coords)
        tile_with_pos = torch.cat([tile_embs, pos_emb], dim=-1)
        tile_embs_fused = self.tile_fusion(tile_with_pos)
        
        # Stage 2: First cross-attention - Glom to WSI tiles
        # This produces X_{G-T} (glom-tile embeddings)
        X_GT = self._apply_glom_tile_attention(
            glom_embs, tile_embs_fused, glom_mask, tile_mask
        )
        
        # Stage 3: Core CT Block - Cross-attention from X_{G-T} to X_L
        # Query: X_{G-T} (glom-tile embeddings) 
        # Key & Value: X_L (lesion embeddings) - same fixed dictionary for all gloms
        # This produces attention matrix A
        ct_output, A = self.cross_attn_ct(
            query=X_GT,  # [B, G, d_m] - glom-tile embeddings
            key=lesion_embeddings,  # [B, L, d_m] - fixed lesion dictionary
            value=lesion_embeddings,  # [B, L, d_m] - fixed lesion dictionary (V matrix)
            return_attention=True 
        )
        
        # Stage 4: Compute γ (lesion relevance scores)
        # γ_l(x) = (1/G) Σ_g α_{gl}
        gamma = self.compute_gamma(A, glom_mask)  # [B, L]
        
        # Stage 5: Compute task-specific logits 
        # For each task t: logit_i^(t) = Σ_l [V·O^(t)]_{l,i} * γ_l(x)
        
        # V is the lesion embeddings (from cross-attention values)
        V = lesion_embeddings  # [B, L, d_m]
        
        # DM-gate head (binary)
        dm_gate_logits, beta_dm_gate = self.compute_logits(
            gamma, V, self.O_dm_gate
        )
        
        # DM-subtype head (3 classes)
        dm_sub_logits, beta_dm_sub = self.compute_logits(
            gamma, V, self.O_dm_sub
        )
        
        # No-DM-subtype head (2 classes)
        no_dm_sub_logits, beta_no_dm_sub = self.compute_logits(
            gamma, V, self.O_no_dm_sub
        )
        
        # Outcome head (binary)
        outcome_logits, beta_outcome = self.compute_logits(
            gamma, V, self.O_outcome
        )
        
        # Stage 6: Compute probabilities
        
        # Gate probability: Pr(z_DM = 1|x)
        pr_dm = torch.sigmoid(dm_gate_logits.squeeze(-1))  # [B]
        
        # Conditional probabilities
        pr_dm_sub = F.softmax(dm_sub_logits, dim=-1)  # [B, 3]
        pr_no_dm_sub = F.softmax(no_dm_sub_logits, dim=-1)  # [B, 2]
        
        # Outcome probability: Pr(Outcome = 1|x)
        pr_outcome = torch.sigmoid(outcome_logits.squeeze(-1))  # [B]
        
        # Stage 7: Compute final 5-category probabilities via gating
        # Pr(y=k|x) = Pr(z_DM=1|x)*Pr(y=k|x,z_DM=1) + Pr(z_DM=0|x)*Pr(y=k|x,z_DM=0)
        final_probs = torch.zeros(B, 5, device=glom_embs.device)
        
        # Non-diabetic categories (controlled by gate)
        pr_no_dm = 1 - pr_dm  # [B]
        final_probs[:, 0] = pr_no_dm * pr_no_dm_sub[:, 0]  # Healthy
        final_probs[:, 1] = pr_no_dm * pr_no_dm_sub[:, 1]  # FSGS-no-DM
        
        # Diabetic categories (controlled by gate)
        final_probs[:, 2] = pr_dm * pr_dm_sub[:, 0]  # DN
        final_probs[:, 3] = pr_dm * pr_dm_sub[:, 1]  # DN+FSGS
        final_probs[:, 4] = pr_dm * pr_dm_sub[:, 2]  # DN+FPE
        
        outputs = {
            # Primary predictions
            'final_probs': final_probs,  # [B, 5] - 5-category classification
            'pr_dm': pr_dm,  # [B] - Pr(diabetes)
            'pr_outcome': pr_outcome,  # [B] - Pr(eGFR decline)
            
            # Conditional probabilities
            'pr_dm_sub': pr_dm_sub,  # [B, 3] - Pr(subtype|DM=1)
            'pr_no_dm_sub': pr_no_dm_sub,  # [B, 2] - Pr(subtype|DM=0)
            
            # Raw logits
            'dm_gate_logits': dm_gate_logits,  # [B, 1]
            'dm_sub_logits': dm_sub_logits,  # [B, 3]
            'no_dm_sub_logits': no_dm_sub_logits,  # [B, 2]
            'outcome_logits': outcome_logits,  # [B, 1]
            
            # Concept bottleneck components (for interpretability)
            'gamma': gamma,  # [B, L] - patient-specific lesion relevance
            'beta_dm_gate': beta_dm_gate,  # [B, L, 1] - lesion importance for DM
            'beta_dm_sub': beta_dm_sub,  # [B, L, 3] - lesion importance for DM subtypes
            'beta_no_dm_sub': beta_no_dm_sub,  # [B, L, 2] - lesion importance for no-DM subtypes
            'beta_outcome': beta_outcome,  # [B, L, 1] - lesion importance for outcome
            
            # Attention matrix 
            'A': A,  # [B, G, L] - attention weights from CT block
        }
        
        return outputs
    
    def _apply_glom_tile_attention(self, glom_embs, tile_embs_fused, glom_mask, tile_mask):
        """
        Apply first cross-attention between glomeruli and WSI tiles
        This produces X_{G-T} (glom-tile embeddings)
        """
        B, G, _ = glom_embs.shape
        T = tile_embs_fused.shape[1]
        
        # Reshape for batch processing
        glom_flat = glom_embs.reshape(B * G, 1, self.glom_dim)
        
        # Expand tiles for each glom
        tile_expanded = tile_embs_fused.unsqueeze(1).expand(-1, G, -1, -1)
        tile_expanded = tile_expanded.reshape(B * G, T, self.tile_dim)
        
        # Expand tile mask if provided
        if tile_mask is not None:
            tile_mask_expanded = tile_mask.unsqueeze(1).expand(-1, G, -1)
            tile_mask_expanded = tile_mask_expanded.reshape(B * G, T)
        else:
            tile_mask_expanded = None
        
        # Apply cross-attention
        glom_tile_out, _ = self.cross_attn_glom_tile(
            query=glom_flat,
            key=tile_expanded,
            value=tile_expanded,
            return_attention=False
        )
        
        # Reshape back
        glom_tile_out = glom_tile_out.view(B, G, self.glom_dim)
        
        # Residual connection and normalization
        glom_enhanced = self.norm1(glom_embs + glom_tile_out)
        glom_enhanced = self.norm2(glom_enhanced + self.ffn_glom_tile(glom_enhanced))
        
        return glom_enhanced
    
    def build_target_attention_H(self, glom_lesion_labels, num_gloms, num_lesions):
        """
        Build target attention matrix H for explanation loss
        
        Args:
            glom_lesion_labels: [B, G, L] - binary indicators of lesion presence
            num_gloms: G
            num_lesions: L
        
        Returns:
            H: [B, G, L] - target attention distribution
        """
        B = glom_lesion_labels.shape[0]
        H = torch.zeros(B, num_gloms, num_lesions, device=glom_lesion_labels.device)
        
        for b in range(B):
            for g in range(num_gloms):
                # Get lesions present in this glom
                present_lesions = glom_lesion_labels[b, g].nonzero(as_tuple=True)[0]
                
                if len(present_lesions) > 0:
                    # Uniform distribution over present lesions
                    H[b, g, present_lesions] = 1.0 / len(present_lesions)
                else:
                    # If no lesions, uniform over all (weak supervision)
                    H[b, g, :] = 1.0 / num_lesions
        
        return H


class ConceptTransformerLoss(nn.Module):
    """
    Loss computation for Concept Transformer
    """
    def __init__(
        self,
        lambda_out: float = 1.0,
        lambda_gate: float = 1.0,
        lambda_sub: float = 1.0,
        lambda_expl: float = 0.5,
    ):
        super().__init__()
        self.lambda_out = lambda_out
        self.lambda_gate = lambda_gate
        self.lambda_sub = lambda_sub
        self.lambda_expl = lambda_expl
        
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict,
        targets: Dict,
        H: Optional[torch.Tensor] = None
    ):
        """
        Compute total loss
        
        L = λ_out*BCE(Outcome) + λ_gate*BCE(z_DM) + λ_sub*[conditional CE] + λ_expl*||A-H||²_F
        
        Args:
            outputs: Model outputs dictionary
            targets: Dictionary with target labels:
                - 'dm_labels': [B] binary (0/1) for diabetes
                - 'outcome_labels': [B] binary for eGFR decline (optional)
                - 'final_labels': [B] indices (0-4) for 5 categories
            H: [B, G, L] target attention distribution
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        device = outputs['pr_dm'].device
        B = outputs['pr_dm'].shape[0]
        
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Outcome loss: λ_out * BCE(Outcome, Pr(Outcome=1|x))
        if 'outcome_labels' in targets:
            outcome_loss = self.bce_loss(
                outputs['pr_outcome'],
                targets['outcome_labels'].float()
            )
            loss_dict['outcome_loss'] = outcome_loss
            total_loss += self.lambda_out * outcome_loss
        
        # 2. Gate loss: λ_gate * BCE(z_DM, Pr(z_DM=1|x))
        if 'dm_labels' in targets:
            gate_loss = self.bce_loss(
                outputs['pr_dm'],
                targets['dm_labels'].float()
            )
            loss_dict['gate_loss'] = gate_loss
            total_loss += self.lambda_gate * gate_loss
        
        # 3. Subtype losses (conditional on gate)
        # λ_sub * [1{z_DM=1}*CE(y, Pr(y|x,z_DM=1)) + 1{z_DM=0}*CE(y, Pr(y|x,z_DM=0))]
        if 'final_labels' in targets:
            final_labels = targets['final_labels']
            
            # Determine which patients are diabetic
            if 'dm_labels' in targets:
                dm_mask = targets['dm_labels']
            else:
                # Infer from final labels (classes 2,3,4 are diabetic)
                dm_mask = (final_labels >= 2).float()
            
            # DM subtype loss (for diabetic patients)
            dm_indices = dm_mask.nonzero(as_tuple=True)[0]
            if len(dm_indices) > 0:
                # Map final labels to DM subtypes: 2->0, 3->1, 4->2
                dm_sub_targets = final_labels[dm_indices] - 2
                dm_sub_loss = self.ce_loss(
                    outputs['dm_sub_logits'][dm_indices],
                    dm_sub_targets
                )
                loss_dict['dm_sub_loss'] = dm_sub_loss
                total_loss += self.lambda_sub * dm_sub_loss
            
            # No-DM subtype loss (for non-diabetic patients)
            no_dm_indices = (1 - dm_mask).nonzero(as_tuple=True)[0]
            if len(no_dm_indices) > 0:
                # Map final labels to no-DM subtypes: 0->0, 1->1
                no_dm_sub_targets = final_labels[no_dm_indices]
                no_dm_sub_loss = self.ce_loss(
                    outputs['no_dm_sub_logits'][no_dm_indices],
                    no_dm_sub_targets
                )
                loss_dict['no_dm_sub_loss'] = no_dm_sub_loss
                total_loss += self.lambda_sub * no_dm_sub_loss
        
        # 4. Explanation loss:# Using Frobenius norm
        if H is not None:
            A = outputs['A']  # [B, G, L] - attention weights from CT block
            # Compute Frobenius norm squared for each patient, then average
            frobenius_per_patient = torch.sum((A - H) ** 2, dim=[1, 2])  # [B]
            expl_loss = frobenius_per_patient.mean()  # Average over batch
            loss_dict['expl_loss'] = expl_loss
            total_loss += self.lambda_expl * expl_loss
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


