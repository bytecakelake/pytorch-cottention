from torch import Tensor
from torch.nn.functional import normalize

def cottention(Q: Tensor, K: Tensor, V: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Computes Cottention.
    
    Reference:
        Mongaras, G., Dohm, T., & Larson, E. C. (2024). 
        Cottention: Linear Transformers With Cosine Attention. 
        arXiv preprint arXiv:2409.18747.
        https://arxiv.org/abs/2409.18747
    
    Args:
        Q (Tensor): Query tensor. (..., sequence dim, feature dim)
        K (Tensor): Key tensor. (..., sequence dim, feature dim)
        V (Tensor): Value tensor. (..., sequence dim, feature dim)
        eps (float): Epsilon for normalization.
        
    Returns:
        Tensor: Attention output.
    """
    Q_norm = normalize(Q, p=2, dim=-1, eps=eps)
    K_norm = normalize(K, p=2, dim=-1, eps=eps)
    attn = Q_norm @ (K_norm.transpose(-2, -1) @ V)
    return attn
