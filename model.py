import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tool import printFeature

class ConvBlock(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, act=None, image_size=256, kernel = 3, group = 1):
        super(ConvBlock, self).__init__()
        if act is None:
            self.act = nn.ReLU()
        self.same = in_channels == out_channels
        self.norm1 = nn.BatchNorm2d(in_channels)
        hidden_dim = out_channels
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        if not self.same:
            self.besame = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        g = 8
        self.conv = nn.Sequential(
            self.norm1,
            act,
            nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel, padding=kernel//2, padding_mode='replicate', groups=group),
            self.norm2,
            act,
            nn.Conv2d(hidden_dim, out_channels, kernel_size=kernel, padding=kernel//2, padding_mode='replicate', groups=group),
        )

    def forward(self, x):
        """
        batchsize, var, time, width, height
        """
        y = self.conv(x)
        if not self.same:
            x = self.besame(x)
        return x + y





class channelAttn(nn.Module):
    def  __init__(self, in_channels=8, act=None, image_size=256,drop=0.1):
        super(channelAttn, self).__init__()
        if act is None:
            self.act = nn.ReLU()
        self.ffv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.ffq = nn.Sequential(
            # nn.Dropout(drop),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            # act,
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
        )
        self.ffk = nn.Sequential(
            # nn.Dropout(drop),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            # act,
            # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
        )
        # self.norm = nn.GroupNorm(in_channels//16, in_channels)
        self.in_channels = in_channels
        self.scale = image_size ** -1
        self.softmax = nn.Softmax(-1)
        self.image_size = image_size

    def forward(self, x):
        """
        batchsize, var*time, width, height
        """
        qkv = x
        # qkv = self.norm(x)
        v = self.ffv(qkv)
        q = self.ffq(qkv)
        k = self.ffk(qkv)
        q = rearrange(q, "b n h w -> b n (h w)")
        k = rearrange(k, "b n h w -> b n (h w)")
        v = rearrange(v, "b n h w -> b n (h w)")
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        v = torch.matmul(attn, v)
        v = rearrange(v, 'b n (h w) -> b n h w', w = self.image_size, h = self.image_size)
        return v+x



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Memo(nn.Module):
    def __init__(self, num_vectors, latent_dim):
        super(Memo, self).__init__()
        self.num_vectors = num_vectors
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_vectors, 1.0 / self.num_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        memodecode_loss = (z_q.detach() - z)**2
        # memodecode_loss = memodecode_loss.mean()

        z_q = z_q.permute(0, 3, 1, 2)
        return {
            "zq": z_q,
            "index":min_encoding_indices,
            "memodecode_loss":memodecode_loss,}

class SE(nn.Module):
    def __init__(self, in_channels=None, act=None):
        super(SE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.l = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            Swish(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        batchsize, var, time, width, height
        """
        w = self.pool(x)
        shape = w.shape
        w = self.l(w.squeeze())
        x = w.reshape(shape) * x
        return x



class downl(nn.Module):
    def __init__(self, in_channels, out_channels, act=None, image_size=None, gnorm =False):
        super().__init__()
        self.convdown = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, padding=1,
                                  stride=2,
                                  padding_mode="replicate")

    def forward(self, x):
        x = self.convdown(x)
        return x


class upl(nn.Module):
    def __init__(self, in_channels, act=None, image_size=None, gnorm =True, lk = 5):
        super().__init__()
        self.convup = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, padding=1, stride=2,
                               groups=in_channels // 8)
        self.norm = nn.GroupNorm(in_channels//16, in_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.convup(x)
        return x


class Fuse(nn.Module):
    def __init__(self, in_channels, kernel=5):
        super().__init__()
        self.norm = nn.GroupNorm(in_channels // 16, in_channels)
        self.l = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2, stride=1,
                               groups=in_channels // 8, padding_mode="replicate")

    def forward(self, x):
        x = self.norm(x)
        x = self.l(x)
        return x


class MemoUnet(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, var_num=7, dropout=0.1, image_size=256,
                 act=Swish(), latent_dim=256, code_num=1024):
        super(MemoUnet, self).__init__()
        self.var_num = var_num
        self.out_channels = out_channels
        self.image_size = image_size
        self.tnum = 5
        downc = [64, 64, 64, 128, 256, latent_dim]
        upc = [[downc[4]+latent_dim, 256],
               [256 + downc[-3], 128],
               [128 + downc[-4], 64],
               [64 + downc[-5], 48],
               [48 + downc[-6]//2, 32],
               ]

        self.timeembed = nn.Sequential(
            nn.LayerNorm(self.tnum),
            nn.Linear(self.tnum, (image_size // 16) * (image_size // 16)),
            Swish(),
            nn.LayerNorm((image_size // 16) * (image_size // 16)),
            nn.Linear((image_size // 16) * (image_size // 16), (image_size // 16) * (image_size // 16)),
        )
        self.memo = Memo(num_vectors=code_num, latent_dim=latent_dim)
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=var_num * in_channels, out_channels=downc[0], kernel_size=3, padding=1,padding_mode="replicate"),
            ConvBlock(in_channels=downc[0], out_channels=downc[0], image_size=image_size, act=act),
        )
        self.down1 = nn.Sequential(
            downl(in_channels=downc[0], out_channels=downc[1]),
            ConvBlock(in_channels=downc[1], out_channels=downc[1], image_size=image_size // 2, act=act),
        )
        self.down2 = nn.Sequential(
            downl(in_channels=downc[1], out_channels=downc[2]),
            ConvBlock(in_channels=downc[2], out_channels=downc[2], image_size=image_size // 4, act=act),
        )
        self.down3 = nn.Sequential(
            downl(in_channels=downc[2], out_channels=downc[3]),
            ConvBlock(in_channels=downc[3], out_channels=downc[3], image_size=image_size // 8, act=act),
        )
        self.down4 = nn.Sequential(
            downl(in_channels=downc[3], out_channels=downc[4]),
        )
        h1 = 6
        self.x4to = nn.Sequential(
            ConvBlock(in_channels=downc[4], out_channels=downc[4], image_size=image_size // 16, act=act),
            channelAttn(in_channels =downc[4], image_size=image_size // 16),
            ConvBlock(in_channels=downc[4], out_channels=downc[4], image_size=image_size // 16, act=act),
        )
        self.to_code = nn.Sequential(
            ConvBlock(in_channels=downc[4], out_channels=downc[4], image_size=image_size // 16, act=act),
            channelAttn(in_channels =downc[4], image_size=image_size // 16),
            ConvBlock(in_channels=downc[4], out_channels=downc[4], image_size=image_size // 16, act=act),
        )
        self.fuset = nn.Conv2d(downc[4] + 1, downc[4], 1, 1, 0)
        self.down4back = nn.Sequential(
            upl(upc[0][0]),
            ConvBlock(in_channels=upc[0][0], out_channels=upc[0][1], image_size=image_size // 8, act=act),
            SE(upc[0][1]),
            ConvBlock(in_channels=upc[0][1], out_channels=upc[0][1], image_size=image_size // 8, act=act),
        )
        self.down3back = nn.Sequential(
            upl(upc[1][0]),
            ConvBlock(in_channels=upc[1][0], out_channels=upc[1][1], image_size=image_size // 4, act=act),
            SE(upc[1][1]),
            ConvBlock(in_channels=upc[1][1], out_channels=upc[1][1], image_size=image_size // 4, act=act),
        )
        self.down2back = nn.Sequential(
            upl(upc[2][0]),
            ConvBlock(in_channels=upc[2][0], out_channels=upc[2][1], image_size=image_size // 2, act=act),
            SE(upc[2][1]),
            ConvBlock(in_channels=upc[2][1], out_channels=upc[2][1], image_size=image_size // 2, act=act),
        )
        self.down1back = nn.Sequential(
            upl(upc[3][0]),
            ConvBlock(in_channels=upc[3][0], out_channels=upc[3][1], image_size=image_size, act=act),
            SE(upc[3][1]),
            ConvBlock(in_channels=upc[3][1], out_channels=upc[3][1], image_size=image_size, act=act),
        )
        self.decode1 = nn.Sequential(
            Fuse(upc[4][0]),
            ConvBlock(in_channels=upc[4][0], out_channels=upc[4][1], image_size=image_size, act=act),
            SE(upc[4][1]),
            ConvBlock(in_channels=upc[4][1], out_channels=upc[4][1], image_size=image_size, act=act),
            ConvBlock(in_channels=upc[4][1], out_channels=out_channels, image_size=image_size, act=act, vali=False),
        )
        self.x3to = nn.Sequential(
            ConvBlock(in_channels=downc[3], out_channels=downc[3], image_size=image_size // 8, act=act),
        )
        self.x2to = nn.Sequential(
            ConvBlock(in_channels=downc[2], out_channels=downc[2], image_size=image_size // 4, act=act),
        )
        self.x1to = nn.Sequential(
            ConvBlock(in_channels=downc[1], out_channels=downc[1], image_size=image_size // 2, act=act),
        )
        self.xto = nn.Sequential(
            ConvBlock(in_channels=downc[0], out_channels=downc[0] // 2, image_size=image_size, act=act),
        )

    def forward(self, x, t):
        """
        batchsize, var, time, width, height
        """
        x = rearrange(x, "b v t w h -> b (v t) w h")
        x = self.encode(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        t = self.timeembed(t.float())
        t = rearrange(t, "b (n w h) -> b n w h", n = 1, h = self.image_size//16, w = self.image_size//16)
        x4 = torch.cat([x4, t], dim = 1)
        x4 = self.fuset(x4)
        x4 = self.x4to(x4)
        xcode = self.to_code(x4)
        memoinfo = self.memo(xcode)
        x4 = torch.cat([x4, memoinfo["zq"]], dim=1)
        x4 = self.down4back(x4)
        x3 = self.x3to(x3)
        x3 = self.down3back(torch.cat([x4, x3], dim=1))
        x2 = self.x2to(x2)
        x2 = self.down2back(torch.cat([x3, x2], dim=1))
        x1 = self.x1to(x1)
        x1 = self.down1back(torch.cat([x2, x1], dim=1))
        x = self.xto(x)
        x = self.decode1(torch.cat([x1, x], dim=1))
        return x, memoinfo
