import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, decoder_dim, embed_dim, vocab_size,
                 encoder_dim=512, dropout_rate=0.5):

        super(Decoder, self).__init__()
        self.encoder_dim = encoder_dim

        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.decode_step = nn.LSTMCell(
            embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer for initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer for initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        # before: (batch_size, encoded_image_size*encoded_image_size, 512)
        mean_encoder_out = encoder_out.mean(dim=1)
        # after: (batch_size, 512)

        # transform 512 (dim image embeddings) in decoder dim
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        #c = h
        return h, c

    def forward(self, word, decoder_hidden_state, decoder_cell_state, encoder_out=None):
        # encoder_out is not used in this forward pass, it's just necessary for the model DecoderWithAttention
        word_embedding = self.embedding(word)
        decoder_hidden_state, decoder_cell_state = self.decode_step(word_embedding, (decoder_hidden_state, decoder_cell_state))
        decoder_hidden_state_drop = self.dropout(decoder_hidden_state)
        scores = self.fc(decoder_hidden_state_drop)
        return scores, decoder_hidden_state, decoder_cell_state 
