import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)

        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask,
                debug=True, alpha_mask=None):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :param Bool debug: Enable debug
        :param Tensor alpha_mask: mask of default enabled values
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, torch.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            if ((mask != 0).nonzero().size(0) == mask.size(1)):
                att[mask] = self.inf[mask]
            else:
                att = att * alpha_mask
        else:
            att = att * alpha_mask
        alpha = F.softmax(att, dim=-1)

        if debug:
            print('alpha_mask',F.softmax(att * alpha_mask))
            print(att)
            print(mask)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

# PYTHONPATH='./qtree' python3 run.py --graph_path ./graph_test/inst_2x2_7_0.txt --max_shape 20  --max_epoch 100  --embd 10  --exp gcn_small_10_5  --save_path ./models_ckpt/  --gcn
