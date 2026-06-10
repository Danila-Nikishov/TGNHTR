import os

# Возвращаем старый, добрый и стабильный Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from spektral.layers import GATConv
from tensorflow.keras import layers


class NoMaskGAT(layers.Layer):
    """
    Тонкая обёртка над GATConv, которая вызывает layer.call() напрямую,
    минуя механизм масок Keras. Это единственный надёжный способ
    избежать передачи mask=[None, None] в GATConv.
    """

    def __init__(self, gat_layer, **kwargs):
        super().__init__(**kwargs)
        self.gat = gat_layer

    def call(self, inputs, training=False):
        # Вызываем .call() напрямую — без __call__, без масок
        return self.gat.call(inputs)

    def build(self, input_shape):
        # Форсируем построение внутреннего слоя
        self.gat.build(input_shape)
        super().build(input_shape)


class TGNHTR(tf.keras.Model):
    def __init__(
        self,
        vocab_size=61,
        node_channels=64,
        output_channels=None,
        attn_heads=8,
        dropout_rate=0.2,
        lstm_units=64,
    ):
        """
        vocab_size:       число символов (без blank); итого классов = vocab_size + 1
        node_channels:    размерность признаков узлов
        output_channels:  если задано явно — переопределяет vocab_size
        """
        super(TGNHTR, self).__init__()

        if output_channels is None:
            output_channels = vocab_size + 1

        # Проекция входных признаков
        self.node_proj = layers.Dense(node_channels)

        # Слои GAT — обёрнуты в NoMaskGAT, чтобы Keras не пробрасывал mask=None
        self.gat1 = NoMaskGAT(
            GATConv(channels=node_channels, attn_heads=attn_heads, concat=True)
        )
        self.gat2 = NoMaskGAT(
            GATConv(channels=node_channels, attn_heads=attn_heads, concat=True)
        )
        self.gat3 = NoMaskGAT(
            GATConv(channels=node_channels, attn_heads=1, concat=False)
        )

        self.dropout = layers.Dropout(dropout_rate)

        # RNN и классификатор
        self.bilstm = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True)
        )
        self.classifier = layers.Dense(output_channels)

    def call(self, inputs, training=False):
        # Ожидаем inputs = [x, a, time_idx]
        x, a, time_idx = inputs

        # Проекция
        x = self.node_proj(x)

        x = self.gat1([x, a])
        x = self.gat2([x, a])
        x = self.gat3([x, a])

        x = self.dropout(x, training=training)

        # Gather: (Batch, Nodes, Features) -> (Batch, Seq_Len, Features)
        x = tf.gather(x, time_idx, batch_dims=1)

        # BiLSTM
        x = self.bilstm(x, training=training)

        # Классификация
        x = self.classifier(x)

        return x

    def get_config(self):
        config = super().get_config()
        return config
