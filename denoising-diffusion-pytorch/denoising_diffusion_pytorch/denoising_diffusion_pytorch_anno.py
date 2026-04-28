import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions
# 检查变量是否存在 (不为 None)，存在返回True，为None返回False
def exists(x):
    return x is not None


def default(val, d):
    """
        获取变量值或返回默认值。

        参数:
        val: 要检查的变量。
        d: 默认值。可以是普通数值，也可以是一个函数(如 lambda)。

        逻辑:
        1. 如果 val 存在(不为None)，直接返回 val。
        2. 如果 val 为 None：
           - 若 d 是函数(callable)，则运行 d() 并返回结果（实现延迟计算，节省性能）。
           - 若 d 是普通值，直接返回 d。
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    """
        将输入值强制转换为元组格式。
        常用于处理模型参数（如 heads, dim_mults），使其既能接受单一数值，也能接受自定义序列。

        参数:
        t: 输入值。可以是任意类型，或者是元组。
        length: 如果输入不是元组，则将其重复成长度为 length 的元组。

        例子:
        cast_tuple(64, length=3)    -> (64, 64, 64)
        cast_tuple((1, 2), length=3) -> (1, 2) (检测到已经是元组，原样返回)
    """
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def divisible_by(numer, denom):
    """
        [功能] 检查一个数能否被另一个数整除
        [术语] numer: numerator (分子), denom: denominator (分母)
        [用途] 例如在 U-Net 中确保图像的长宽能被下采样因子整除，防止跳跃连接时特征图尺寸不匹配
    """
    return (numer % denom) == 0


def identity(t, *args, **kwargs):
    """
        [功能] 恒等函数：原封不动地返回输入值 t。
        [术语] identity: 恒等/一致；args: arguments (位置参数)；kwargs: keyword arguments (关键字参数)。
        [用途] 作为占位符使用。当逻辑上需要一个函数作为参数，但实际不需要对数据做处理时（如：可选的归一化步骤），
              传入此函数可保证流程通畅。
    """
    return t


def cycle(dl):
    """
        [功能] 将dl转换为无限循环的生成器。
        [术语] cycle: 循环；dl: DataLoader (数据加载器)；yield: 产出 (生成器关键字)。
        [解释]
        1. Python 函数可以没有 return。但含有 yield 的函数不再是普通函数，而是生成器。
        2. yield 会在产出数据后暂停函数执行，并在下次请求数据时从暂停处继续。
        3. 此处配合 while True 使用，可确保在整个模型训练期间，数据能够源源不断地循环读取。
        4. 当内层的 for 循环把比如 1000 张图全部吐完（遍历完毕）的那一瞬间，外层的 while True 会立刻生效，把程序重新拨回到下一个epoch
        5. 因此，这个函数永远不会自然停止。在训练代码中，我们通过指定总步数 (steps) 来决定何时手动关闭这个“流水线”
    """
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    """
    [功能] 检查数字是否具有整数平方根（即是否为完全平方数）。
    [术语] int: integer (整数); squareroot: 平方根。
    [用途] 确保生成图像的总数可以拼成一个正方形矩阵（如 25 张拼成 5x5）。
    """
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    """
    [功能] 将一个总数拆分成若干个指定大小的组。
    [术语] divisor: 除数; remainder: 余数; groups: 完整组数。
    [例子] num_to_groups(25, 8) -> [8, 8, 8, 1]。
    [用途] 显存管理。当需要生成大量图像时，将其拆分为多个显存可承受的小批次处理。
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    """
    [功能] 统一图像的色彩模式（如将所有图转为 RGB）。
    [术语] convert: 转换; img_type: 目标图像模式(如 'RGB', 'L'); fn: function (函数缩写)。
    [逻辑]
    1. 检查输入图像 image 的当前模式是否与目标 img_type 一致。
    2. 若不一致，调用 Pillow 库的 convert 方法进行强制转换，确保后续进入模型的数据维度统一。
    3. 这样做可以防止因数据集包含 RGBA 或灰度图而导致的通道数不匹配报错。
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    """
    [功能] 将图像像素值从 [0, 1] 映射到 [-1, 1]。
    [术语] normalize: 归一化; neg: negative (负的)。
    [理由] 扩散模型在 [-1, 1] 的数据分布下训练效果最好，这有助于保持去噪过程中的数值稳定性。
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """
    [功能] 反归一化：将数据从 [-1, 1] 映射回 [0, 1]。
    [术语] unnormalize: 反归一化。
    [用途] 采样结束准备保存图片时使用。因为 Pillow 等库保存图片需要数据在 [0, 1] 或 [0, 255] 之间。
    """
    return (t + 1) * 0.5


# small helper modules

def Upsample(dim, dim_out=None):
    """
    [功能] 上采样模块：将图像的长宽放大 2 倍。
    [参数]
        dim: 输入通道数。
        dim_out: 输出通道数。若不传(None)，则通过 default 函数自动设为与输入相同。
    [术语] scale_factor: 缩放倍数；default: 备选判定函数。
    [逻辑]
    1. nn.Upsample 以 2 倍率(scale_factor=2)放大尺寸。
    2. nn.Conv2d 随后通过 3x3 卷积核调整特征，其输出维度由 default(dim_out, dim) 确定。
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        # 如果用户没指定输出维度，卷积层就维持原有的通道数
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    """
    [功能] 下采样模块：通过“空间换深度”将图像长宽缩小一倍。
    [术语] Rearrange: 重新排列张量维度；p1, p2: 像素分块大小。
    [逻辑]
    1. Rearrange 将 2x2 的空间像素块平铺到通道维度。
       输入 (H, W) -> 输出 (H/2, W/2)，通道数由 C 变为 C*4。
    2. 这种做法比池化(Pooling)更优秀，因为它保留了所有原始像素信息。
    3. nn.Conv2d(dim * 4, ...) 接着使用 1x1 卷积将翻了 4 倍的通道数调整回目标维度。
    """
    return nn.Sequential(
        # 这一步是核心：把空间上的 2x2 区域“折叠”进通道
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        # 调整厚度，默认将 dim*4 压缩回 dim
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5  # self.scale是个标量
        """
        [参数说明] self.g (Gain / 增益参数)
        1. 使用 torch.ones 初始化为全 1，保证训练初期不改变原始特征强度。
        2. 形状 (1, dim, 1, 1) 是为了匹配 BCHW 格式，利用广播机制让 
           每个通道拥有一个独立的可学习缩放系数。
        3. nn.Parameter 把它注册为模型的成员变量，确保此权重会被加入优化器进行训练更新。
        """
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        """
        [公式拆解] output = normalize(x) * g * scale
        1. F.normalize(x, dim=1): 在通道维做归一化，把每个像素的特征向量长度统一。
        简单例子：
            假设一个像素点的形状是 (1, 3, 1, 1)，值分别是 [a, b, c]。
            计算范数：norm = sqrt(a² + b² + c²)
            归一化：[a/norm, b/norm, c/norm]
        2. * self.g: 触发广播机制，应用 (1, dim, 1, 1) 的可学习权重，调整各通道重要性。
        3. * self.scale: 乘以根号 dim 的标量，用于恢复因归一化而损失的数值增益(Gain)。
        """
        return F.normalize(x, dim=1) * self.g * self.scale


# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    """
    [功能] 正弦位置编码器：将“时间步”数字转换为高维特征向量。
    [理由] 神经网络很难直接理解“5”或“100”这类数字的含义，
          通过正弦/余弦变换，可以将时间点映射为一组独特的周期信号，
          使得模型能辨别当前处于去噪的哪个阶段。
    [术语] theta: 周期衰减系数；half_dim: 特征向量的一半长度。
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        # 计算不同通道对应的频率缩放比例
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # 将时间步 x 扩展维度，并与频率相乘（触发广播机制）
        # 结果是一个 [batch, half_dim] 的矩阵
        emb = x[:, None] * emb[None, :]

        # 拼接 sin 和 cos，形成最终的时间嵌入向量 [batch, dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(Module):
    """
    这是上面的进阶版 不在固定为10000
    [功能] 随机或可学习的傅里叶特征位置编码。
    [灵感] 借鉴 @crowsonkb 在 v-diffusion-jax 中的实现，取代传统的 10000 衰减公式。
    [优势] 允许模型自己学习最适合当前数据集的频率分布，而不是依赖人工设定的超参数。
    """

    def __init__(self, dim, is_random=False):
        super().__init__()
        # 确保维度是偶数，因为要平分给 sin 和 cos
        assert divisible_by(dim, 2)
        half_dim = dim // 2

        # 核心：用高斯白噪声初始化频率权重。
        # 如果 is_random=False (默认)，则开启 requires_grad，模型将在训练中自适应更新这些频率！
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        # x 形状由 (b,) 变为 (b, 1)
        x = rearrange(x, 'b -> b 1')

        # 将时间 x 与频率权重相乘 (触发广播)，并乘以 2π 映射到弧度制
        # freqs 形状: (b, half_dim)
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi

        # 计算 sin 和 cos 并拼接，fouriered 形状变为 (b, dim)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)

        # 注意：最后把原始时间标量 x 也拼接到前面，最终输出形状为 (b, dim + 1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

# building block modules (构建块模块)

class Block(Module):
    """
    [功能] U-Net 的基础构建块，融合了特征提取、归一化与条件注入。
    [结构] 卷积 (Conv2d) -> 归一化 (RMSNorm) -> 调制 (FiLM) -> 激活 (SiLU) -> Dropout
    """

    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        # 使用 3x3 卷积提取空间特征，padding=1 保持空间分辨率不变
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        # 使用之前定义的均方根归一化，提升训练稳定性
        self.norm = RMSNorm(dim_out)
        # 使用平滑的 SiLU(Swish) 激活函数，在生成任务中表现优于 ReLU
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        # 1. 基础图像特征处理
        x = self.proj(x)
        x = self.norm(x)

        # 2. 条件注入 (Condition Injection / FiLM)
        # scale_shift 由时间步 (Time Step) (或之后的stable diffusion的文本提示 Prompt) 经过全连接层映射而来。
        if exists(scale_shift):
            scale, shift = scale_shift
            # x * (scale + 1) + shift:
            # 通过拉伸(scale)和平移(shift)图像特征，告诉当前层处于去噪的哪个阶段。
            # 加 1 是为了让初始状态 (scale趋近0时) 保持类似恒等映射 (Identity)。
            x = x * (scale + 1) + shift

        # 3. 非线性激活与防过拟合
        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    """
    [功能] 带有时间条件注入的残差网络块，是 U-Net 的核心构成单元。
    [结构] 包含两个基础 Block，一个处理时间的 MLP，以及一条残差(跳跃)连接。
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        # 时间转译器：将时间向量的维度放大两倍，为了后续切分为 scale 和 shift
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        # 核心图像处理模块
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)

        # 残差连接对齐器：如果输入输出通道不同，用 1x1 卷积调整；否则直接放行
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        # 【时间注入准备阶段】
        if exists(self.mlp) and exists(time_emb):
            # 1. 经过 MLP 得到加倍的通道
            time_emb = self.mlp(time_emb)
            # 2. 变形为 (batch, channel, 1, 1) 触发后续 FiLM 的广播机制
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # 3. 将厚度切成两半，打包成 (scale, shift) 元组
            scale_shift = time_emb.chunk(2, dim=1)

        # 【图像处理与融合阶段】
        # 将原始图像和时间条件(scale_shift)送入第一个 block
        h = self.block1(x, scale_shift=scale_shift)
        # 经过第二个 block 进一步提取特征
        h = self.block2(h)

        # 【残差相加】
        # H(x) = F(x) + x : 输出 = 变化量 + 原始输入，防止网络退化和梯度消失
        return h + self.res_conv(x)


class LinearAttention(Module):
    """
    [宏观总结]
    这是一个“省显存版”的注意力机制。
    因为高清图片的像素太多，标准的 Attention 算不过来会崩溃。
    这个类通过玩了一个数学魔术（先算 K 和 V），成功把计算量降了下来，让破电脑也能跑得动。
    """

    def __init__(
            self,
            dim,  # 输入特征的原始厚度 (比如 512)
            heads=4,  # 4 个头（工作小组）
            dim_head=32,  # 每个小组只分配 32 个核心维度去算
            num_mem_kv=4  # 准备 4 张“场外援助”记忆卡
    ):
        super().__init__()
        self.scale = dim_head ** -0.5  # 缩放因子：防止后面算出来的数字大到爆炸
        self.heads = heads

        # 内部实际工作的总厚度 (比如 4 * 32 = 128)
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)  # 归一化：把数据压平稳，防止训练崩溃

        # 制作“公共记忆卡片”，给模型提供一些通用的参考信息。
        # 形状：2份(给K和V), 4个小组, 每张卡厚度32, 4张卡
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))

        # 生产 QKV 的机器：一次性把特征加厚 3 倍，方便后面一刀切成 3 份
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 出口机器：算完之后，把 128 维压缩回原来的 512 维
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape  # 拿到图片的 批次、厚度、高、宽

        x = self.norm(x)  # 数据先过一遍归一化维稳

        # 【第一步：生产 QKV】
        # 把图片加厚 3 倍，然后顺着厚度方向切 3 份，变成 Q(查询), K(键), V(值)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # 【第二步：格式转换】
        # 把 QKV 里的特征拆给 4 个小组，并把 2D 的图片拉直成 1D 的像素长条
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # 【第三步：挂载记忆卡 (场外援助)】
        # 1. 按照图片数量 b，把原本只有 1 套的记忆卡复印 b 份
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        # 2. 把复印好的记忆卡，硬拼接到 K 和 V 的资料库里 (相当于加了几页参考书)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        # 【第四步：特征归一化】
        # 这个版本特有的操作，提前对 Q 和 K 做 Softmax 处理，压制极端数值
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        # 【第五步：核心计算 (省显存黑魔法)】
        # 传统是 Q 和 K 先乘，这里为了省钱，先让 K 和 V 乘，算出一个“精华压缩包” (context)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 然后拿 Q 去跟这个“精华压缩包”相乘，得到最终的注意力提取结果
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        # 【第六步：还原收尾】
        # 把刚才拉直的 1D 长条重新折叠回 2D 图片的高(x)和宽(y)，把 4 个小组合并回总厚度
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        # 走出口机器，恢复初始的特征厚度，下班交接给下一层
        return self.to_out(out)


class Attention(Module):
    """
    标准的 Scaled Dot-Product 多头注意力机制。
    空间/时间复杂度为 O(N^2)，通常用于 U-Net 的深层（低分辨率特征图）。
    """

    def __init__(
            self,
            dim,  # 输入特征维度
            heads=4,  # 多头数量
            dim_head=32,  # 单个头的特征维度
            num_mem_kv=4,  # 附加的可学习键/值对数量
            flash=False  # 是否使用底层 FlashAttention 算子加速
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads  # 注意力计算的内部总维度

        self.norm = RMSNorm(dim)

        # 封装的注意力计算模块 (负责 QK^T/sqrt(d) * V)
        self.attend = Attend(flash=flash)

        # 引入全局可学习参数，形状为 (2, heads, num_mem_kv, dim_head)
        # 索引 0 用于 Key，索引 1 用于 Value
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

        # 将输入映射为 Q, K, V，输出通道数为 hidden_dim 的 3 倍
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 线性投影层，恢复至原始维度
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape  # 拿到图片的 批次、厚度、高、宽

        x = self.norm(x)  # 数据过安检维稳

        # 【第一步：生产 QKV】
        # 把图片加厚 3 倍，然后顺着厚度方向切 3 份，变成 Q(查询), K(键), V(值)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # 【第二步：格式转换 (拉直图片)】
        # 把原本平面的图片 (x, y) 强行拉直成一维的像素长条 (x y)
        # 把厚度拆给 4 个工作小组，最后分别贴上 q, k, v 的标签
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        # 【第三步：挂载记忆卡 (场外援助)】
        # 1. 按照传入的图片数量 b，把原本只有 1 套的记忆卡复印 b 份
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        # 2. 把复印好的记忆卡，硬拼接到 K 和 V 的资料库最前面 (让资料库变长了4个单位)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        # 【第四步：核心计算 (丢进黑盒)】
        # 最省心的一步！不用自己手写复杂的公式了。
        # 直接把 q, k, v 扔进智能计算器，它会自动完成全图像素的互相查询，并吐出结果。
        out = self.attend(q, k, v)

        # 【第五步：还原收尾】
        # 把刚才拉直的 1D 长条，重新像叠被子一样折叠回 2D 图片的高(x)和宽(y)
        # 把 4 个工作小组的成果合并回总厚度
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        # 走出口
        return self.to_out(out)


# model

class Unet(Module):
    """
    [架构说明]
    适用于扩散模型的 U-Net 架构，负责执行单步的噪声预测（去噪计算）。
    输入：带噪图像 x_t (及可选的自条件 x_0) 与时间步 t。
    输出：预测的噪声张量 \epsilon (或包含方差预测)。
    """

    def __init__(
            self,
            dim,  # 基础特征维度 (通道数基准)
            init_dim=None,  # 初始卷积的输出通道数
            out_dim=None,  # 最终输出的通道数 (通常为 3)
            dim_mults=(1, 2, 4, 8),  # 每一层特征维度的膨胀倍率列表
            channels=3,  # 输入图像的物理通道数 (RGB=3)
            self_condition=False,  # 是否拼接上一步预测的 x_0 作为条件输入
            learned_variance=False,  # 是否预测方差 (决定输出维度是 C 还是 2C)
            learned_sinusoidal_cond=False,  # 是否使用可学习的时间编码
            random_fourier_features=False,  # 是否使用随机傅里叶特征编码时间
            learned_sinusoidal_dim=16,  # 可学习时间编码的维度
            sinusoidal_pos_emb_theta=10000,  # 正弦位置编码的基频
            dropout=0.,  # 正则化：随机失活率
            attn_dim_head=32,  # 注意力机制：单头的特征维度
            attn_heads=4,  # 注意力机制：多头数量
            full_attn=None,  # 列表：指定哪一层使用标准注意力 (O(N^2)复杂度)
            flash_attn=False  # 是否调用底层 FlashAttention 算子加速
    ):
        super().__init__()

        # ==========================================
        # 1. 确定初始维度与通道数映射
        # ==========================================
        self.channels = channels
        self.self_condition = self_condition

        # 若开启自条件，输入通道数翻倍 (按通道维度拼接带噪图像与预测图像)
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # 初始特征提取层：使用 7x7 大卷积核获取较大的初始感受野
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        # 计算 U-Net 每一层的特征维度。例如：[64, 64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        # 构建下采样阶段的进出维度配对。例如：[(64, 64), (64, 128), ...]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ==========================================
        # 2. 时间步嵌入 (Time Embeddings) 构建
        # ==========================================
        time_dim = dim * 4  # 时间特征向量的最终维度，通常放大以增强信号

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # 选择时间编码映射基类
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        # 时间特征处理 MLP：将标量 t 映射为高维连续特征向量，供后续 FiLM 层使用
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ==========================================
        # 3. 注意力机制策略与参数对齐
        # ==========================================
        # 若未指定，则默认仅在特征分辨率最小的最深层开启标准注意力机制
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)

        # 使用 cast_tuple 将单一参数扩展为与层数对应的元组，便于后续迭代分配
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults), "注意力配置列表长度必须与网络层数一致"

        # ==========================================
        # 4. 预配置基础计算模块 (偏函数)
        # ==========================================
        # 提前绑定全局参数，简化后续实例化代码
        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # ==========================================
        # 5. 构建 U-Net 编码器 (Downs)
        # ==========================================
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        # 遍历维度配对列表，逐层构建下采样模块
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            # 根据配置选择 O(N) 的线性注意力或 O(N^2) 的标准注意力
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        # ==========================================
        # 6. 构建 U-Net 瓶颈层 (Mid)
        # ==========================================
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        # 瓶颈层特征分辨率最低，默认强制使用标准注意力提取全局依赖
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # ==========================================
        # 7. 构建 U-Net 解码器 (Ups)
        # ==========================================
        # 逆序遍历配置列表进行上采样构建
        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                # [关键] dim_out + dim_in：接收同层编码器的跳跃连接(Skip Connection)，通道数相加
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        # ==========================================
        # 8. 最终输出映射层
        # ==========================================
        # 若预测方差，输出通道数需翻倍
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # 最后一层解码器的输入包含来自第一层编码器的跳跃连接，因此输入维度为 init_dim * 2
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        # 1x1 卷积将高维特征投影至物理图像空间的通道数
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)


@property
def downsample_factor(self):
    return 2 ** (len(self.downs) - 1)


def forward(self, x, time, x_self_cond=None):
    """
    [前向传播说明]
    定义数据在 U-Net 中的单次计算图。
    输入张量形状假定为 (B, C_in, H, W)，其中 C_in 通常为 3。
    """

    # ==========================================
    # 0. 尺寸安全检查
    # ==========================================
    # 确保输入图像的高宽能够被网络最大下采样倍率整除（避免上采样时尺寸无法对齐）
    assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), \
        f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

    # ==========================================
    # 1. 输入处理与自条件融合
    # ==========================================
    if self.self_condition:
        # 如果未传入历史预测值 x_0，则使用全零张量占位
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        # 在通道维度 (dim=1) 拼接：(B, 3, H, W) + (B, 3, H, W) -> (B, 6, H, W)
        x = torch.cat((x_self_cond, x), dim=1)

    # 跨界大门：将物理通道映射为基础特征维度 dim
    # 形状变化：(B, C_in, H, W) -> (B, dim, H, W)
    x = self.init_conv(x)

    # 全局跳跃连接备份：保存未经任何压缩的初始高维特征，留给网络最末端兜底
    r = x.clone()

    # ==========================================
    # 2. 时间信号解码
    # ==========================================
    # 将时间步标量 t 映射为连续的高维特征向量
    # t 的形状：(B,) -> (B, time_dim)
    t = self.time_mlp(time)

    # ==========================================
    # 3. 编码器 (Encoder) - 下采样阶段
    # ==========================================
    h = []  # 宏观跳跃连接的“栈”(Stack)，用于缓存下采样过程中的中间特征

    # 遍历配置好的层，例如 4 层
    for block1, block2, attn, downsample in self.downs:
        # 3.1 第一级残差提取 (注入时间向量 t)
        x = block1(x, t)
        h.append(x)  # 第一次入栈：缓存 block1 的输出特征

        # 3.2 第二级残差提取 (注入时间向量 t)
        x = block2(x, t)

        # 3.3 局部注意力机制与局部残差连接 (微观操作)
        x = attn(x) + x  # F(x) + x，特征维度 C 保持不变

        h.append(x)  # 第二次入栈：缓存 block2 + attn 后的输出特征

        # 3.4 空间下采样 (尺寸减半，维度通常翻倍)
        # 形状变化举例：(B, C, H, W) -> (B, 2C, H/2, W/2)
        x = downsample(x)

    # ==========================================
    # 4. 瓶颈层 (Bottleneck) - 谷底阶段
    # ==========================================
    # 此时张量处于极小空间分辨率、极大特征宽度的状态 (例如 8x8, 512维)
    x = self.mid_block1(x, t)
    x = self.mid_attn(x) + x  # 重型全注意力捕捉全局关联 + 局部残差
    x = self.mid_block2(x, t)

    # ==========================================
    # 5. 解码器 (Decoder) - 上采样阶段
    # ==========================================
    for block1, block2, attn, upsample in self.ups:
        # 5.1 第一次宏观跳跃拼接 (Skip Connection)
        # 弹出同层 block2 存入的特征图。维度 C 瞬间翻倍
        x = torch.cat((x, h.pop()), dim=1)
        x = block1(x, t)  # 通过残差块将翻倍的通道融合压缩回预定维度

        # 5.2 第二次宏观跳跃拼接
        # 弹出同层 block1 存入的特征图。维度 C 再次翻倍
        x = torch.cat((x, h.pop()), dim=1)
        x = block2(x, t)

        # 5.3 局部注意力与局部残差
        x = attn(x) + x

        # 5.4 空间上采样 (尺寸翻倍，维度通常减半)
        # 形状变化举例：(B, C, H, W) -> (B, C/2, 2H, 2W)
        x = upsample(x)

    # ==========================================
    # 6. 终极融合与输出映射
    # ==========================================
    # 将网络末端特征与最开头的全局备份 r 拼接
    x = torch.cat((x, r), dim=1)

    # 最后一次残差块特征深加工
    x = self.final_res_block(x, t)

    # 1x1 卷积：将网络特征维度拍扁回物理空间的通道数 (例如 dim -> 3)
    # 最终形状：(B, out_dim, H_original, W_original)
    return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(Module):
    """
    [核心定位]
    DDPM 扩散模型的“数学引擎与车间调度”。
    负责：预计算所有带根号的物理常数、执行前向加噪、计算损失、执行逆向去噪循环。
    """

    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,  # 总步数 T
            sampling_timesteps=None,  # [可忽略] 用于 DDIM 加速采样的步数
            objective='pred_noise',  # 预测目标，原版 DDPM 默认预测噪声
            beta_schedule='sigmoid',  # beta 方差的增长曲线
            schedule_fn_kwargs=dict(),
            ddim_sampling_eta=0.,  # [可忽略] DDIM 的随机性系数
            auto_normalize=True,  # 是否自动将图像映射到 [-1, 1]
            offset_noise_strength=0.,  # [可忽略] 解决图像太亮/太暗的补丁 (2023)
            min_snr_loss_weight=False,  # [可忽略] 损失函数加权平衡策略 (2023)
            min_snr_gamma=5,  # [可忽略] 同上
            immiscible=False  # [可忽略] 不相混扩散策略 (2023)
    ):
        super().__init__()
        # [安检] 确保 U-Net 的输入输出通道一致
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model  # 传入的 U-Net
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition  # [可忽略] 自条件补丁

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(
            image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        # [可忽略] 'pred_v' 是 Imagen 论文提出的参数化技巧，基础只看 'pred_noise'
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}

        # 1. 生成 beta 序列 (每一步的方差)
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        # 2. 核心数学预计算：算出 alphas 和 累乘的 alphas_cumprod (\bar{\alpha}_t)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 对应公式里的 \bar{\alpha}_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # 对应 \bar{\alpha}_{t-1}

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # [可忽略] DDIM 采样参数配置
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # [可忽略] GPU 底层管理：将变量注册进显存，且不参与梯度更新
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # ==========================================
        # 3. 前向加噪公式系数 (DDPM Eq. 4)
        # 目标：x_t = sqrt(\bar{\alpha}_t) * x_0 + sqrt(1 - \bar{\alpha}_t) * \epsilon
        # ==========================================
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # [可忽略] 以下几个是在不同预测目标下反推 x_0 时用到的变形系数
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # ==========================================
        # 4. 后验推断公式系数 (DDPM Eq. 6 & 7)
        # 目标：算出 q(x_{t-1} | x_t, x_0) 的真实均值和方差
        # ==========================================
        # Eq. 7: 真实的后验方差 \tilde{\beta}_t
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # 为了数值稳定，对极小的方差做截断求对数
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

        # Eq. 7: 真实的后验均值 \tilde{\mu}_t 里的两个常数系数
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # [可忽略] 2023年的改进项
        self.immiscible = immiscible
        self.offset_noise_strength = offset_noise_strength
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        # 5. 根据预测目标，设置 Loss 的时间步权重
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)  # 标准 DDPM
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # 归一化开关
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        # [可忽略] 获取当前模型所在的 GPU 设备
        return self.betas.device

    # ==========================================
    # 数学推导辅助函数区
    # ==========================================
    def predict_start_from_noise(self, x_t, t, noise):
        # [核心] 根据当前带噪图 x_t 和预测出的噪声，通过公式反推出粗糙的原图 x_0
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        # [反推] 如果模型直接预测了 x_0，用它反推出对应的噪声
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # [可忽略] v 预测模式的两个专属函数
    def predict_v(self, x_start, t, noise):
        ...

    def predict_start_from_v(self, x_t, t, v):
        ...

    def q_posterior(self, x_start, x_t, t):
        # [核心] DDPM Eq. 6 & 7：将算出的 x_0 和现在的 x_t 代入，求出理论上上一步 (x_{t-1}) 的均值和方差
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        # 让 U-Net 出马，预测结果
        model_output = self.model(x, t, x_self_cond)
        # 将粗糙推导出的 x_0 强行裁切到 [-1, 1] 之间，防止颜色失真爆炸
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            # [可忽略] 裁切原图后，重新修正噪声值
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        # [可忽略] elif 'pred_x0' ... elif 'pred_v' ...

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        # [核心采样第一步] 根据模型预测，结合后验公式，算出如果往回走一步，分布的均值和方差是多少
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()  # [可忽略] 底层加速，生成时关闭梯度计算
    def p_sample(self, x, t: int, x_self_cond=None):
        # [核心采样第二步] DDPM Algorithm 2
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)

        # 拿到均值和方差
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=True)

        # 制造纯随机扰动 z
        noise = torch.randn_like(x) if t > 0 else 0.  # 最后一步 t=0 时不需要加噪声了

        # x_{t-1} = \mu + \sigma * z
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        # [核心] 从纯噪声生成图像的 1000 步倒序循环
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)  # 纯高斯噪声 x_T
        imgs = [img]

        x_start = None

        # 倒计时循环：999, 998 ... 0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None  # [可忽略] 自条件
            img, x_start = self.p_sample(img, t, self_cond)  # 走一步去噪
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        # 反归一化回 [0, 1] 以便保存图片
        ret = self.unnormalize(ret)
        return ret

    # [可忽略] def ddim_sample(...) DDIM 的加速跳步采样逻辑，数学公式完全不同

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        # 对外的生成接口，自动帮你选跑普通的循环还是 DDIM 的循环
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps=return_all_timesteps)

    # [可忽略] def interpolate(...) 隐空间插值，用来做两张生成图片之间的丝滑过渡
    # [可忽略] def noise_assignment(...) 配合 immiscible 特性的功能

    @autocast('cuda', enabled=False)  # [可忽略] 混合精度加速
    def q_sample(self, x_start, t, noise=None):
        # [核心] DDPM Algorithm 1 的加噪操作
        # 一步到位！不需要循环，直接算出第 t 步的图
        noise = default(noise, lambda: torch.randn_like(x_start))

        # [可忽略] if self.immiscible: ...

        # x_t = sqrt(\bar{\alpha}_t) * x_0 + sqrt(1-\bar{\alpha}_t) * \epsilon
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None):
        # [核心] 训练网络的核心步骤
        b, c, h, w = x_start.shape

        # 1. 抽个随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))

        # [可忽略] offset noise 补丁逻辑 ...

        # 2. 调用加噪函数，一键生成第 t 步的烂图 x
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # [可忽略] x_self_cond 相关的自条件逻辑 ...
        x_self_cond = None

        # 3. 让 U-Net 上手，看烂图猜出刚刚加的噪声
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise  # 目标就是最初始抽取的那个真实 noise
        # [可忽略] elif 'pred_x0' ... elif 'pred_v' ...

        # 4. 计算均方误差 MSE Loss
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 乘上损失函数的权重系数
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        # [入口] 整个类在训练前向传播时的第一入口
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'

        # 核心动作：随机掷骰子，抽取一个范围在 0 到 T(1000) 之间的时间步 t
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 归一化到 [-1, 1]
        img = self.normalize(img)

        # 丢进损失计算车间
        return self.p_losses(img, t, *args, **kwargs)


# dataset classes

class Dataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_horizontal_flip=False,
            convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# trainer class

class Trainer:
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            augment_horizontal_flip=True,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            convert_image_to=None,
            calculate_fid=True,
            inception_block_idx=2048,
            max_grad_norm=1.,
            num_fid_samples=50000,
            save_best_and_latest_only=False
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (
                       train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip,
                          convert_image_to=convert_image_to)

        assert len(
            self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming." \
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                         nrow=int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
