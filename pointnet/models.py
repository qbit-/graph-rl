from torch import nn
import torch
import  torch.nn.functional as F

from pointnet.attention import Attention


class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir=False, rnn=True):
        """
        Initiate Encoder
        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        if rnn:
            self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden):
        """
        Encoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim, hidden_dim):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)
        self.decoder = nn.LSTMCell(embedding_dim, hidden_dim)
        self.mask = nn.Parameter(torch.ones(1), requires_grad=False)
        self.runner = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context,
                add_mask=None,
                greedy=True,
                debug=False):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :param Tensor add_mask: mask of inputs
        :param Bool greedy: Enable greedy decoding
        :param Bool debug: Enable debug
        :return: (Output probabilities, actions)
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # if mask is None:
        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        # debug = False
        # Recurrence loop
        for i in range(input_length):
            if i >= 1:
                debug = False
            h_t, c_t = self.decoder(decoder_input, hidden)
            hidden_t, outs = self.att(h_t, context, torch.eq(mask, 0), debug, add_mask)
            h_t = torch.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            # h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)
            # masked_outs = (outs*add_mask)
            normed_outs = (outs + 1e-8)*add_mask
            normed_outs = normed_outs / normed_outs.sum(dim=-1)[..., None]

            iter_tens = torch.tensor(i, dtype=torch.float).to(mask)
            new_mask = mask * torch.ge(add_mask.sum(-1),iter_tens ).unsqueeze(1).float().to(mask)
            masked_outs = normed_outs * new_mask
            dist = torch.distributions.Categorical(probs=masked_outs+1e-8)
            if greedy:
                # Get maximum probabilities and indices
                max_probs, actions = masked_outs.max(1)
            else:
                actions = dist.sample()
            if debug:
                print(mask)
                print(masked_outs)
                print(new_mask)

            one_hot_pointers = (runner == actions.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask = mask * (1 - one_hot_pointers)
            # add_mask = add_mask* (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)
            # outputs.append(outs.unsqueeze(0))
            outputs.append(dist.log_prob(actions).unsqueeze(0))
            pointers.append(actions.unsqueeze(1))

        # ( batch x seq_len x seq_len (as a vocab) )
        pointers = torch.cat(pointers, 1)
        outputs = torch.cat(outputs).transpose(0,1)

        return (outputs, pointers), hidden

