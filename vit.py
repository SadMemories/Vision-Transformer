import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbed(nn.Module):

    def __init__(self, image_size, patch_size, hid_size, in_c=3):
        super(PatchEmbed, self).__init__()
        self.image_size = image_size
        self.proj = nn.Conv2d(in_c, hid_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: batch x 3 x H x W
        assert x.shape[2] == self.image_size and x.shape[3] == self.image_size, \
                "input shape is not equal {}".format(self.image_size)

        out = self.proj(x).flatten(2).transpose(-2, -1)
        return out


class Attention(nn.Module):

    def __init__(self, hid_size, heads):
        super(Attention, self).__init__()
        self.heads = heads
        self.hid_size = hid_size

        self.qkv = nn.Linear(hid_size, 3 * hid_size)
        self.proj = nn.Linear(hid_size, hid_size)

    def forward(self, x):  # x: batch x num_patch x hid_size
        assert self.hid_size % self.heads == 0, "hid_size is not devisible by heads"

        batch, N, C = x.shape[0], x.shape[1], x.shape[2]
        qkv = self.qkv(x).reshape(batch, N, 3, self.heads, self.hid_size // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hid_size)
        scores_soft = F.softmax(scores, -1)
        v_score = torch.matmul(scores_soft, v).transpose(1, 2).reshape(batch, N, C)

        return self.proj(v_score)


class MLP(nn.Module):

    def __init__(self, hid_size, mlp_size, act_layer):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(hid_size, mlp_size)
        self.fc2 = nn.Linear(mlp_size, hid_size)
        self.act_layer = act_layer()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act_layer(out)
        out = self.fc2(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, hid_size, mlp_size, norm_layer, heads, act_layer, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm1 = norm_layer(hid_size)
        self.norm2 = norm_layer(hid_size)

        self.attn = Attention(hid_size, heads)
        self.mlp = MLP(hid_size, mlp_size, act_layer)

    def forward(self, x):
        out = x + self.attn(self.norm1(x))
        fin_out = out + self.mlp(self.norm2(out))

        return fin_out


class VisionTransformer(nn.Module):

    def __init__(self, patch, image_size, hid_size, layers, heads, num_classes,
                 mlp_ratio=4.0, norm_layer=None, act_layer=None):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(image_size, patch, hid_size)
        self.num_patch = (image_size // patch) * (image_size // patch)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.hid_size = hid_size

        self.pos_embed = nn.Parameter(torch.zeros((1, self.num_patch+1, hid_size)))
        self.cls_token = nn.Parameter(torch.zeros((1, 1, hid_size)))
        self.head = nn.Linear(hid_size, num_classes)
        self.norm = norm_layer(hid_size)

        self.blocks = nn.Sequential(*[
            TransformerBlock(hid_size, int(hid_size*mlp_ratio), norm_layer, heads, act_layer)
            for _ in range(layers)
        ])

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weight)

    def forward(self, x):  # x: batch x 3 x H x W
        batch = x.shape[0]
        patch_embed = self.patch_embed(x)  # batch x N x hid_size

        cls_token = self.cls_token.expand((batch, 1, self.hid_size))
        inp_embed = torch.cat([cls_token, patch_embed], dim=1)
        inp_embed = inp_embed + self.pos_embed

        out = self.blocks(inp_embed)
        out = self.norm(out)  # out = self.norm(out[:, 0])
        out = self.head(out[:, 0])

        return out


def _init_weight(m):

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def vit_base_patch16_224(num_classes):

    vit = VisionTransformer(patch=16, image_size=224, hid_size=768,
                            layers=12, heads=12, num_classes=num_classes)
    return vit


def main():
    model = vit_base_patch16_224(15)
    inp = torch.randn((16, 3, 224, 224))
    out = model(inp)
    print(out.shape)


if __name__ == '__main__':
    main()