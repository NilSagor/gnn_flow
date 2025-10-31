import torch 
import torch.nn as nn




class Resnet(nn.Module):
    def __init__(self, num_blocks, c_hidden, block_class, act_fn):
        ...

        blocks = []
        for block_idx, block_count in enumerate(num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0)
                blocks.append(
                    block_class(
                        c_in = c_hidden[block_idx if not subsample else (block_idx -1)],
                        c_out = c_hidden[block_idx],
                        act_fn = act_fn,
                        subsample = subsample,
                    )
                )

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear()
        )

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x


