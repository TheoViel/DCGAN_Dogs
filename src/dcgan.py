import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GenBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2, padding=1, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        self.conv = nn.ConvTranspose2d(
            c_in, c_out, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return self.relu(x)


class DCGanGenerator(nn.Module):
    def __init__(
        self,
        n=128,
        latent_dim=128,
        nb_classes=120,
        embedding_dim=32,
        use_bn=True,
        use_embed=False,
    ):
        super().__init__()

        self.use_embed = use_embed
        self.embedding_dim = embedding_dim if use_embed else 0
        self.embedding = torch.nn.Embedding(nb_classes, embedding_dim)

        self.cnn = nn.Sequential(
            GenBlock(
                latent_dim + self.embedding_dim,
                n * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                use_bn=use_bn,
            ),  # 8n x 4 x 4
            GenBlock(
                n * 8, n * 4, kernel_size=4, stride=2, padding=1, use_bn=use_bn
            ),  # 4n x 8 x 8
            GenBlock(
                n * 4, n * 2, kernel_size=4, stride=2, padding=1, use_bn=use_bn
            ),  # 2n x 16 x 16
            GenBlock(
                n * 2, n, kernel_size=4, stride=2, padding=1, use_bn=use_bn
            ),  # n x 32 x 32
            nn.ConvTranspose2d(n, 3, 4, 2, 1, bias=False),  # 3 x 64 x 64
            nn.Tanh(),
        )

    def forward(self, noise, label=None):
        if self.use_embed:
            noise = torch.cat(
                (noise, self.embedding(label).view(-1, self.embedding_dim, 1, 1)), 1
            )
        return self.cnn(noise)


class DisBlock(nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size=4, stride=2, padding=1, slope=0.2, use_bn=True
    ):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return self.lrelu(x)


class DCGanDiscriminator(nn.Module):
    def __init__(
        self,
        n=128,
        nb_classes=120,
        embedding_dim=32,
        nb_ft=64,
        slope=0.2,
        use_embed=False,
        use_ft_matching=False,
        use_bn=True,
    ):

        super().__init__()

        self.use_embed = use_embed
        self.use_ft_matching = use_ft_matching
        self.nb_ft = nb_ft if (use_embed or use_ft_matching) else 1

        self.cnn = nn.Sequential(
            DisBlock(
                3, n, kernel_size=4, stride=2, padding=1, slope=slope, use_bn=False
            ),  # n x 32 x 32
            DisBlock(
                n, 2 * n, kernel_size=4, stride=2, padding=1, slope=slope, use_bn=use_bn
            ),  # 2n x 16 x 16
            DisBlock(
                2 * n,
                4 * n,
                kernel_size=4,
                stride=2,
                padding=1,
                slope=slope,
                use_bn=use_bn,
            ),  # 4n x 8 x 8
            DisBlock(
                4 * n,
                8 * n,
                kernel_size=4,
                stride=2,
                padding=1,
                slope=slope,
                use_bn=use_bn,
            ),  # 8n x 4 x 4
            nn.Conv2d(
                n * 8, self.nb_ft, kernel_size=4, stride=1, padding=0, bias=False
            ),  # nb_ft x 1 x 1
        )

        # embed
        self.embedding = torch.nn.Embedding(nb_classes, embedding_dim)
        self.dense = nn.Linear(nb_ft + embedding_dim, 1)
        self.dense_classes = nn.Linear(self.nb_ft + embedding_dim, nb_classes)

    def forward(self, imgs, label=None, return_classes=False):
        features = self.cnn(imgs).view(-1, self.nb_ft)

        if self.use_embed and label is not None:
            embed = self.embedding(label)
            x = torch.cat((features, embed), 1)

            out = self.dense(x)
            out_classes = self.dense_classes(x) if return_classes else 0
        else:
            out = features
            out_classes = 0

        return out, out_classes, features


if __name__ == "__main__":
    import numpy as np

    noise = torch.from_numpy(np.random.random(size=(1, 128, 1, 1))).float()

    gen = DCGanGenerator()
    # print(gen)

    img = gen(noise)
    # print(img.size())

    dis = DCGanDiscriminator()
    # print(dis)

    p = dis(img)
    # print(p)
