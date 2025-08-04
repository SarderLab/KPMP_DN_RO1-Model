import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    """Cross-attention between query and key-value pairs"""
    def __init__(self, dim_q, dim_kv, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.head_dim = dim_q // num_heads
        
        assert self.dim_q % self.num_heads == 0, "dim_q must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, key_padding_mask=None):
        B, N_q, _ = query.shape
        N_kv = key.shape[1]
        
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.dim_q)
        out = self.out_proj(out)
        
        return out, attn_weights


class LesionEncoder(nn.Module):
    """Learnable lesion encoder for generating lesion-specific embeddings"""
    def __init__(self, num_lesions=2, embed_dim=768):
        super().__init__()
        # Learnable embeddings for each lesion type
        self.lesion_embeddings = nn.Parameter(torch.randn(num_lesions, embed_dim))
        
        # Transform to generate final lesion encodings
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, batch_size):
        """Generate lesion encodings for the batch"""
        lesion_encodings = self.encoder(self.lesion_embeddings)
        return lesion_encodings.unsqueeze(0).expand(batch_size, -1, -1)


class PatientLevelMultiTaskModel(nn.Module):
    """
    Multi-task model for patient-level kidney pathology analysis
    
    Tasks:
    1. Individual glomeruli lesion classification
    2. Patient-level lesion probability estimation  
    3. Diabetic nephropathy (DN) prediction
    
    Key components:
    - Cross-attention between glomeruli and WSI tiles for spatial context
    - Learnable lesion encoder for disease-specific representations
    - Multi-task heads with weighted loss combination
    """
    
    def __init__(self, 
                 glom_dim=768,
                 tile_dim=1536, 
                 num_lesion_classes=2,
                 num_heads=8,
                 hidden_dim=1024,
                 dropout=0.1,
                 multi_label=False):
        super().__init__()
        
        self.glom_dim = glom_dim
        self.tile_dim = tile_dim
        self.num_lesion_classes = num_lesion_classes
        self.multi_label = multi_label
        
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
        
        # Cross-attention: Glom to WSI tiles
        self.cross_attn_glom_tile = CrossAttentionBlock(
            dim_q=glom_dim,
            dim_kv=tile_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Learnable lesion encoder
        self.lesion_encoder = LesionEncoder(
            num_lesions=num_lesion_classes,
            embed_dim=glom_dim
        )
        
        # Cross-attention: Glomeruli to Lesion encodings
        self.cross_attn_glom_lesion = CrossAttentionBlock(
            dim_q=glom_dim,
            dim_kv=glom_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward networks
        self.ffn1 = self._build_ffn(glom_dim, hidden_dim, dropout)
        self.ffn2 = self._build_ffn(glom_dim, hidden_dim, dropout)
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(glom_dim)
        self.norm2 = nn.LayerNorm(glom_dim)
        self.norm3 = nn.LayerNorm(glom_dim)
        self.norm4 = nn.LayerNorm(glom_dim)
        
        # Task-specific prediction heads
        self._build_prediction_heads(glom_dim, dropout)
        
    def _build_ffn(self, dim, hidden_dim, dropout):
        """Build feed-forward network"""
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def _build_prediction_heads(self, glom_dim, dropout):
        """Build task-specific prediction heads"""
        # 1. Individual glomeruli lesion classifier
        if self.multi_label:
            self.glom_lesion_classifier = nn.Sequential(
                nn.Linear(glom_dim, glom_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(glom_dim // 2, self.num_lesion_classes),
                nn.Sigmoid()  # Multi-label
            )
        else:
            self.glom_lesion_classifier = nn.Sequential(
                nn.Linear(glom_dim, glom_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(glom_dim // 2, self.num_lesion_classes) 
            )
        
        # 2. Patient-level aggregation
        self.glom_pooling = nn.Sequential(
            nn.Linear(glom_dim, glom_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Patient-level lesion probability estimator
        self.patient_lesion_classifier = nn.Sequential(
            nn.Linear(glom_dim, glom_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(glom_dim // 2, self.num_lesion_classes),
            nn.Sigmoid()  # Probability output
        )
        
        # 4. DN prediction components
        self.lesion_relevance_scorer = nn.Sequential(
            nn.Linear(glom_dim, self.num_lesion_classes),
            nn.Sigmoid()
        )
        
        # Learnable lesion importance weights for DN
        self.lesion_coefficients = nn.Parameter(torch.randn(self.num_lesion_classes))
        
        # Final DN predictor
        self.dn_predictor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, glom_embs, tile_embs, coords, glom_mask=None, tile_mask=None):
        """
        Forward pass
        
        Args:
            glom_embs: [B, G, glom_dim] - Glom embeddings
            tile_embs: [B, T, tile_dim] - WSI tile embeddings  
            coords: [B, T, 2] - Tile coordinates
            glom_mask: [B, G] - Mask for padded glomeruli
            tile_mask: [B, T] - Mask for padded tiles
            
        Returns:
            Dictionary with all predictions and intermediate features
        """
        B, G, _ = glom_embs.shape
        
        # Stage 1: Tile processing with positional encoding
        pos_emb = self.pos_encoder(coords)
        tile_with_pos = torch.cat([tile_embs, pos_emb], dim=-1)
        tile_embs_fused = self.tile_fusion(tile_with_pos)
        
        # Stage 2: Glomeruli-WSI cross-attention
        glom_enhanced = self._glom_tile_attention(
            glom_embs, tile_embs_fused, glom_mask, tile_mask
        )
        
        # Stage 3: Glomeruli-lesion cross-attention  
        glom_final = self._glom_lesion_attention(glom_enhanced, B, G)
        
        # Stage 4: Multi-task predictions
        return self._compute_predictions(glom_final, glom_mask)
    
    def _glom_tile_attention(self, glom_embs, tile_embs_fused, glom_mask, tile_mask):
        """Apply cross-attention between glomeruli and WSI tiles"""
        B, G, _ = glom_embs.shape
        
        # Reshape for cross-attention (each glom attends to all tiles)
        glom_flat = glom_embs.view(B * G, 1, self.glom_dim)
        tile_expanded = tile_embs_fused.unsqueeze(1).expand(-1, G, -1, -1)
        tile_expanded = tile_expanded.reshape(B * G, -1, self.tile_dim)
        
        # Expand tile masks
        if tile_mask is not None:
            tile_mask_expanded = tile_mask.unsqueeze(1).expand(-1, G, -1)
            tile_mask_expanded = tile_mask_expanded.reshape(B * G, -1)
        else:
            tile_mask_expanded = None
        
        # Cross-attention
        glom_tile_out, _ = self.cross_attn_glom_tile(
            query=glom_flat,
            key=tile_expanded,
            value=tile_expanded,
            key_padding_mask=tile_mask_expanded
        )
        
        # Reshape back and apply residual + normalization
        glom_tile_out = glom_tile_out.view(B, G, self.glom_dim)
        glom_enhanced = self.norm1(glom_embs + glom_tile_out)
        glom_enhanced = self.norm2(glom_enhanced + self.ffn1(glom_enhanced))
        
        return glom_enhanced
    
    def _glom_lesion_attention(self, glom_enhanced, B, G):
        """Apply cross-attention between glomeruli and lesion encodings"""
        # Get lesion encodings
        lesion_encodings = self.lesion_encoder(B)
        
        # Reshape for cross-attention
        glom_flat = glom_enhanced.view(B * G, 1, self.glom_dim)
        lesion_expanded = lesion_encodings.unsqueeze(1).expand(-1, G, -1, -1)
        lesion_expanded = lesion_expanded.reshape(B * G, self.num_lesion_classes, self.glom_dim)
        
        # Cross-attention
        glom_lesion_out, _ = self.cross_attn_glom_lesion(
            query=glom_flat,
            key=lesion_expanded,
            value=lesion_expanded
        )
        
        # Reshape back and apply residual + normalization
        glom_lesion_out = glom_lesion_out.view(B, G, self.glom_dim)
        glom_final = self.norm3(glom_enhanced + glom_lesion_out)
        glom_final = self.norm4(glom_final + self.ffn2(glom_final))
        
        return glom_final
    
    def _compute_predictions(self, glom_final, glom_mask):
        """Compute all task predictions"""
        # Task 1: Individual glomeruli lesion classification
        glom_lesion_logits = self.glom_lesion_classifier(glom_final)
        
        # Patient-level aggregation (masked mean pooling)
        if glom_mask is not None:
            mask_expanded = (~glom_mask).float().unsqueeze(-1)
            glom_pooled = (glom_final * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            glom_pooled = glom_final.mean(dim=1)
        
        glom_pooled = self.glom_pooling(glom_pooled)
        
        # Task 2: Patient-level lesion probability estimation
        patient_lesion_logits = self.patient_lesion_classifier(glom_pooled)
        
        # Task 3: DN prediction
        lesion_relevance_scores = self.lesion_relevance_scorer(glom_final)
        
        # Aggregate lesion scores across glomeruli
        if glom_mask is not None:
            mask_expanded = (~glom_mask).float().unsqueeze(-1)
            masked_scores = lesion_relevance_scores * mask_expanded
            overall_lesion_logits = masked_scores.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            overall_lesion_logits = lesion_relevance_scores.mean(dim=1)
        
        # Weighted combination using learnable lesion coefficients
        lesion_coefficients_norm = torch.softmax(self.lesion_coefficients, dim=0)
        dn_score = torch.sum(overall_lesion_logits * lesion_coefficients_norm.unsqueeze(0), dim=1, keepdim=True)
        dn_probability = self.dn_predictor(dn_score)
        
        return {
            'glom_lesion_logits': glom_lesion_logits,           # Individual glom predictions
            'patient_lesion_logits': patient_lesion_logits,     # Patient lesion probabilities
            'dn_probability': dn_probability,                   # DN prediction
            'lesion_relevance_scores': lesion_relevance_scores, # Per-glom lesion scores
            'overall_lesion_logits': overall_lesion_logits,     # Aggregated lesion scores
            'lesion_coefficients': lesion_coefficients_norm,    # Learned lesion importance
            'glom_features': glom_final,                        # Final glom features
            'patient_features': glom_pooled,                    # Patient-level features
        }