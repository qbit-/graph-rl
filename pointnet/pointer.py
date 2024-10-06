from simple_models import *


import torch.nn as nn

from pointnet.models import Encoder, Decoder


class PointerNet(nn.Module):
    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False, gcn=None):
        """
        Initiate Pointer-Net
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir

        self.embedding = gcn if gcn is not None else nn.Linear(2, embedding_dim)
        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
        # self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = nn.Parameter(torch.FloatTensor(embedding_dim))

        nn.init.uniform_(self.decoder_input0, -(1. / np.sqrt(self.embedding_dim)), (1. / np.sqrt(self.embedding_dim)))

    def forward(self, inputs, features, mask=None, greedy=True, debug=False):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence
        :param Tensor mask: Default enabled actions
        :param Bool greedy: Enable greedy decoding
        :param Bool debug: Enable debug
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)
        # embd_dom = inputs.size(2)

        decoder_input = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # inputs = inputs.view(batch_size, input_length, -1)

        embedded_inputs = self.embedding(features, inputs).view(batch_size, input_length, -1)

        # encoder_outputs, (hidden, context) = self.encoder(embedded_inputs)

        encoder_hidden = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hiddens = self.encoder(embedded_inputs,
                                                       encoder_hidden)

        if self.bidir:
            decoder_hidden = (torch.cat(encoder_hidden[0][-2:], dim=-1),
                               torch.cat(encoder_hidden[1][-2:], dim=-1))
        else:
            decoder_hidden = (encoder_hiddens[0][-1],
                               encoder_hiddens[1][-1])

        (outputs, pointers), decoder_hiddens = self.decoder(embedded_inputs,
                                                           decoder_input,
                                                           decoder_hidden,
                                                           encoder_outputs,
                                                           add_mask=mask,
                                                           greedy=greedy,
                                                           debug=debug)

        return outputs, pointers
