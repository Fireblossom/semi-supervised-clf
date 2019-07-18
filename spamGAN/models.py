import torch
import torch.nn as nn
import texar as tx

class Generator(nn.Module):
    """Generator wrapper for checkpointing"""

    def __init__(self, vocab_size, decoder_config, dropout):
        super(Generator, self).__init__()
        self.decoder = tx.modules.BasicRNNDecoder(vocab_size=vocab_size,
                                                  hparams=decoder_config,
                                                  cell_dropout_mode=dropout)


class RNNDiscriminator(nn.Module):
    """Discriminator wrapper"""

    def __init__(self, disc_config, dropout):
        super(RNNDiscriminator, self).__init__()
        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            hparams=disc_config['encoder'], cell_dropout_mode=dropout)


class RNNClassifier(nn.Module):
    def __init__(self, class_config, dropout):
        super(RNNClassifier, self).__init__()
        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            hparams=class_config['encoder'], cell_dropout_mode=dropout)


class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_config):
        super(Embedder, self).__init__()
        self.embedder = tx.modules.WordEmbedder(vocab_size=vocab_size,
                                                hparams=emb_config)


