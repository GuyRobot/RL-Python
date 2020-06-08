import torch.nn as nn
import torch


class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes,
                 dropout=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout),
            nn.Softmax(dim=1)
        )

    def __call__(self, x, *args, **kwargs):
        return self.pipe(x)


if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    v = torch.tensor([[2, 3]], dtype=torch.float32)
    out = net(v)
    print(net)
    print(out)
