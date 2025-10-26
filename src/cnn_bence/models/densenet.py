import torch 
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        super(DenseLayer, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, bn_size*growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size*growth_rate),
            act_fn(),
            nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out
    
class DenseBlock(nn.Module):
    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
        super(DenseBlock, self).__init__()
        layers = []
        for layer_idx in range(num_layers):
            layers.append(DenseLayer(
                c_in = c_in + layer_idx*growth_rate,
                bn_size = bn_size,
                growth_rate=growth_rate,
                act_fn=act_fn
            ))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.blocks(x)
        return out
    

class TransitionLayer(nn.Module):
    def __init__(self, c_in, c_out, act_fn):
        super(TransitionLayer, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=1),
            nn.AvgPool2d((2, 2))
        )

    def forward(self, x):
        out = self.net(x)
        return out
    

class DenseNet(nn.Module):
    def __init__(self, num_classes, num_layers, bn_size, growth_rate, act_fn):
        super(DenseNet, self).__init__()
        c_hidden = bn_size * growth_rate

        self.input_net = nn.Sequential(
            3, c_hidden, kernel_size=1, bias = False
        )

        blocks = []
        for block_idx, num_layer in enumerate(num_layers):
            blocks.append(
                DenseBlock(
                    c_in = c_hidden + num_layer*growth_rate,
                    bn_size = bn_size,
                    growth_rate = growth_rate,
                    act_fn = act_fn
                )
            )
            c_hidden = c_hidden + num_layer*growth_rate
            if block_idx < len(num_layers) -1:
                blocks.append(
                    TransitionLayer(
                        c_in = c_hidden,
                        c_out = c_hidden // 2,
                        act_fn = act_fn
                    )
                )
                c_hidden = c_hidden//2

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.BatchNorm2d(c_hidden),
            act_fn(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear(c_hidden, num_classes)
        )


    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
        