import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, emb_dims, num_cont_features, lin_layer_sizes,
                 output_size, emb_dropout, lin_layer_dropouts):
        """
        :param emb_dims: list of two element tuples
            One tuple for each categorical feature - the first elem of
            a tuple denotes the number of unique values of the feature,
            the second one denotes the embedding dim to be used for that
            feature.
        :param num_cont_features: integer
            The number of continuous features in the data.
        :param lin_layer_sizes: list of integers
            The size of each linear layer.
        :param output_size: integer
        :param emb_dropout: float
            The dropout to be used after the embedding layer.
        :param lin_layer_dropouts: list of floats
            The dropouts to be used after each lin layer.
        """
        super(FFNN, self).__init__()

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        self.num_embs = sum([y for _, y in emb_dims])
        self.num_cont_features = num_cont_features

        head_lin_layer = nn.Linear(self.num_embs + self.num_cont_features,
                                   lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList(
            [head_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                                for i in range(len(lin_layer_sizes) - 1)]
        )

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)

        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm layers
        self.head_bn_layer = nn.BatchNorm1d(self.num_cont_features)
        self.tail_bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes]
        )

        # Dropout layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.dropout_layer = nn.ModuleList(
            [nn.Dropout(size) for size in lin_layer_dropouts]
        )

    def forward(self, cont_data, cat_data):
        assert self.num_embs + self.num_cont_features > 0

        if self.num_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.num_cont_features != 0:
            norm_cont_data = self.head_bn_layer(cont_data)

            if self.num_embs != 0:
                x = torch.cat((x, norm_cont_data), dim=1)
            else:
                x = norm_cont_data

        for lin_layer, bn_layer, dropout_layer in \
                zip(self.lin_layers, self.tail_bn_layers, self.dropout_layer):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x
