import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, a, b, pooling):
        super(Block, self).__init__()
        self.pooling = pooling

        # conv path a: two 3x3 convs
        self.conva1 = nn.Conv2d(in_channels, a, kernel_size=3, padding=1)
        self.bn_a1 = nn.BatchNorm2d(a)
        self.conva2 = nn.Conv2d(a, b, kernel_size=3, padding=1)
        self.bn_a2 = nn.BatchNorm2d(b)

        # conv path b: two 5x5 convs
        self.convb1 = nn.Conv2d(in_channels, a, kernel_size=5, padding=2)
        self.bn_b1 = nn.BatchNorm2d(a)
        self.convb2 = nn.Conv2d(a, b, kernel_size=5, padding=2)
        self.bn_b2 = nn.BatchNorm2d(b)

        # conv path c: one 1x1 conv
        self.convc = nn.Conv2d(in_channels, b, kernel_size=1, padding=0)
        self.bn_c = nn.BatchNorm2d(b)

        # conv path d: 3x3 then 1x1 conv
        self.convd1 = nn.Conv2d(in_channels, a, kernel_size=3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(a)
        self.convd2 = nn.Conv2d(a, b, kernel_size=1, padding=0)
        self.bn_d2 = nn.BatchNorm2d(b)

        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # path a
        a = F.relu(self.bn_a1(self.conva1(x)))
        a = F.relu(self.bn_a2(self.conva2(a)))
        if self.pooling:
            a = self.pool(a)

        # path b
        b = F.relu(self.bn_b1(self.convb1(x)))
        b = F.relu(self.bn_b2(self.convb2(b)))
        if self.pooling:
            b = self.pool(b)

        # path c
        c = F.relu(self.bn_c(self.convc(x)))
        if self.pooling:
            c = self.pool(c)

        # path d
        d = F.relu(self.bn_d1(self.convd1(x)))
        d = F.relu(self.bn_d2(self.convd2(d)))
        if self.pooling:
            d = self.pool(d)

        # concatenate along channel dimension
        out = torch.cat([a, b, c, d], dim=1)
        return out


class UNetPlusInception(nn.Module):
    def __init__(self, img_rows=224, img_cols=224, depth=3):
        super(UNetPlusInception, self).__init__()
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.depth = depth

        # Encoder
        self.conv1_1 = nn.Conv2d(depth, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # BatchNorm layers for encoder convs
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.bn5_2 = nn.BatchNorm2d(512)

        # Inception-like blocks
        self.block1 = Block(depth, 16, 16, pooling=False)
        self.block2 = Block(16 * 4, 32, 32, pooling=True)  # 4 paths concatenated
        self.block3 = Block(32 * 4, 64, 64, pooling=True)
        self.block4 = Block(64 * 4, 128, 128, pooling=True)

        # Decoder with ConvTranspose2d (upsampling)
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(256 + 256 + 128 * 4, 256, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.bn6_2 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(128 + 128 + 64 * 4, 128, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.bn7_2 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(64 + 64 + 32 * 4, 64, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.bn8_2 = nn.BatchNorm2d(64)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(32 + 32 + 16 * 4, 32, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.bn9_2 = nn.BatchNorm2d(32)

        self.conv10 = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = F.relu(self.bn1_1(self.conv1_1(x)))
        c1 = F.relu(self.bn1_2(self.conv1_2(c1)))
        p1 = self.pool1(c1)

        c2 = F.relu(self.bn2_1(self.conv2_1(p1)))
        c2 = F.relu(self.bn2_2(self.conv2_2(c2)))
        p2 = self.pool2(c2)

        c3 = F.relu(self.bn3_1(self.conv3_1(p2)))
        c3 = F.relu(self.bn3_2(self.conv3_2(c3)))
        p3 = self.pool3(c3)

        c4 = F.relu(self.bn4_1(self.conv4_1(p3)))
        c4 = F.relu(self.bn4_2(self.conv4_2(c4)))
        p4 = self.pool4(c4)

        c5 = F.relu(self.bn5_1(self.conv5_1(p4)))
        c5 = F.relu(self.bn5_2(self.conv5_2(c5)))

        # Inception blocks
        xx1 = self.block1(x)
        xx2 = self.block2(xx1)
        xx3 = self.block3(xx2)
        xx4 = self.block4(xx3)

        # Decoder
        up6 = self.up6(c5)
        up6 = torch.cat([up6, c4, xx4], dim=1)
        c6 = F.relu(self.bn6_1(self.conv6_1(up6)))
        c6 = F.relu(self.bn6_2(self.conv6_2(c6)))

        up7 = self.up7(c6)
        up7 = torch.cat([up7, c3, xx3], dim=1)
        c7 = F.relu(self.bn7_1(self.conv7_1(up7)))
        c7 = F.relu(self.bn7_2(self.conv7_2(c7)))

        up8 = self.up8(c7)
        up8 = torch.cat([up8, c2, xx2], dim=1)
        c8 = F.relu(self.bn8_1(self.conv8_1(up8)))
        c8 = F.relu(self.bn8_2(self.conv8_2(c8)))

        up9 = self.up9(c8)
        up9 = torch.cat([up9, c1, xx1], dim=1)
        c9 = F.relu(self.bn9_1(self.conv9_1(up9)))
        c9 = F.relu(self.bn9_2(self.conv9_2(c9)))

        out = self.conv10(c9)
        out = self.sigmoid(out)
        return out


# Dice coefficient and loss for PyTorch
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
