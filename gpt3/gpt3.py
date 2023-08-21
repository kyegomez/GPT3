from torch.nn import nn
from gpt3.model import Transformer, Decoder, AutoregressiveWrapper



class GPT3(nn.Module):
    def __init__(self,
                 num_tokens=50477,
                 max_seq_len=4096,
                 dim=12288,
                 depth=96,
                 heads=96,
                 attn_dim_head=128):
        super().__init__()

        self.model = Transformer(
            num_tokens=self.num_tokens,
            max_seq_len=self.max_seq_len,
            attn_layers = Decoder(
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
            )
        ).cuda()

        self.model = AutoregressiveWrapper(self.model)
    def forward(self, text, **kwargs):
        return self.model(text)
