import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from model.original_capsule import CapsuleLayer


class CapsuleNet(nn.Module):
    def __init__(self, num_classes):
        super(CapsuleNet, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=self.num_classes, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * self.num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices.data)

        # reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))

        return classes, reconstructions
