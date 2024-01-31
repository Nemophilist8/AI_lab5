import torch.nn as nn

d_k = d_v = 64

# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,device,d_ff=512):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)  # [batch_size, seq_len, d_model]

class CoAttentionEncoderLayer(nn.Module):
    def __init__(self, embed_dim, device):
        super(CoAttentionEncoderLayer, self).__init__()
        self.enc_self_attn1 = nn.MultiheadAttention(embed_dim=embed_dim , num_heads=8)
        self.pos_ffn1 = PoswiseFeedForwardNet(d_model=embed_dim, device=device)
        self.enc_self_attn2 = nn.MultiheadAttention(embed_dim=embed_dim , num_heads=8)
        self.pos_ffn2 = PoswiseFeedForwardNet(d_model=embed_dim, device=device)

    def forward(self, enc_inputs1, enc_inputs2):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs1, attn1 = self.enc_self_attn1(enc_inputs2, enc_inputs1, enc_inputs1)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs1 = self.pos_ffn1(enc_outputs1)
        # enc_outputs: [batch_size, src_len, d_model]

        enc_outputs2, attn2 = self.enc_self_attn2(enc_inputs1, enc_inputs2, enc_inputs2)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs2 = self.pos_ffn2(enc_outputs2)
        return enc_outputs1, attn1, enc_outputs2, attn2