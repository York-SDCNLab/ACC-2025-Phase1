import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F

from .mlp import MLP

class MultiEncoder(nn.Module):
    def __init__(
        self,
        include_image: bool = True,
        include_vector: bool = True
    ):
        super().__init__()
        self.include_image = include_image
        self.include_vector = include_vector

        if self.include_image:
            self.image_encoder = ConvEncoder(in_channels = 3, depth=32)
        else:
            self.image_encoder = None

        if self.include_vector:
            self.vector_encoder = DenseEncoder(
                in_dim=5,
                out_dim = 64,
                hidden_dim = 400,
                hidden_layers = 2
            )

            '''self.duration_encoder = DenseEncoder(
                in_dim=1,
                out_dim = 16,
                hidden_dim = 64,
                hidden_layers = 1
            )'''

            '''self.hardware_encoder = DenseEncoder(
                in_dim=10,
                out_dim = 64,
                hidden_dim = 400,
                hidden_layers = 2
            )

            self.waypoint_encoder = DenseEncoder(
                in_dim=200,
                out_dim = 64,
                hidden_dim = 400,
                hidden_layers = 2
            )'''
        else:
            self.vector_encoder = None

        combined_dim = ((self.image_encoder.out_dim if self.image_encoder else 0) + 
                        (self.vector_encoder.out_dim if self.vector_encoder else 0)
                        #(self.duration_encoder.out_dim if self.vector_encoder else 0)
                        #(self.hardware_encoder.out_dim if self.vector_encoder else 0) + 
                        #(self.waypoint_encoder.out_dim if self.vector_encoder else 0)
                        )
        
        self.out_dim = 512

        self.mlp = MLP(combined_dim, self.out_dim, 400, 2, nn.LayerNorm, nn.ReLU)

        
        
    def forward(self, obs) -> torch.Tensor:
        embeds = []

        #get image embeddings
        if self.image_encoder:
            T, B, _, _, _ = obs["image"].shape
            image = obs["image"] #T, B, 3, h, w
            #image = flatten_batch(obs["image"], nonbatch_dims=3)[0]

            #embed_image = torch.zeros((T, B, self.image_encoder.out_dim), device=image.device)

            #if valid.any():
            #    embed_image[valid] = self.image_encoder.forward(image[valid])
            
            embed_image = self.image_encoder.forward(image)
            embeds.append(embed_image)

        #get vector observation embeddings
        if self.vector_encoder:
            T, B, _ = obs["state"].shape

            #state = torch.tensor([2.182, -0.087, 1.57, 0.0, 0.0], device = obs["state"].device)
            embed_vecobs = self.vector_encoder(obs["state"][..., :5].to(torch.float32))
            embeds.append(embed_vecobs)

            #duration_vecobs = self.duration_encoder(obs["fsm_duration"].to(torch.float32))
            #embeds.append(duration_vecobs)

            '''embed_hardware = self.hardware_encoder(obs["hardware_metrics"].to(torch.float32))
            embeds.append(embed_hardware)

            #waypoints are only valid if we have a gps signal
            waypoints = obs["waypoints"].reshape(T, B, -1) #T, B, N*2
            valid = obs["gps_valid"] #T, B
            embed_waypoints = torch.zeros((T, B, self.waypoint_encoder.out_dim), device=waypoints.device)

            if valid.any():
                embed_waypoints[valid] = self.waypoint_encoder.forward(waypoints[valid].to(torch.float32))

            embeds.append(embed_waypoints)'''

        embed = torch.cat(embeds, dim=-1)
        embed = self.mlp(embed)

        return embed
    
class DenseEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim = 64,
        hidden_dim = 400,
        hidden_layers = 2,
        activation = nn.ELU
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            activation()
        ]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-3),
                activation()
            ]
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y

class BasicBlock(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        out_planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        shortcut = []
        if stride != 1:
            shortcut = [
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            ]

        self.shortcut = nn.Sequential(*shortcut)

        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation((self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 32,
        activation = nn.ELU
    ):
        super().__init__()
        self.out_dim = 512 #16896 #8960
        
        #for 320x240
        self.conv_rgb = nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.bn_rgb = nn.BatchNorm2d(4)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(32)

        #bottleneck
        self.conv_bottle1 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.bottle_bn1 = nn.BatchNorm2d(32)
        self.conv_bottle2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bottle_bn2 = nn.BatchNorm2d(32)
        self.conv_bottle3 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.bottle_bn3 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(32*40*30, self.out_dim)
        self.activation = nn.ReLU()

    def make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(self.in_planes, stride)]
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        n_step, n_batch, n_channel, width, height = x.shape

        x = x.reshape(n_step*n_batch, n_channel, width, height)
        x = self.activation(self.bn_rgb(self.conv_rgb(x)))

        up = self.up(x)

        y1 = self.activation(self.bn1(self.conv1(up)))
        y2 = self.activation(self.bn2(self.conv2(y1)))
        y3 = self.activation(self.bn3(self.conv3(y2)))
        y4 = self.activation(self.bn4(self.conv4(y3)))

        y5 = self.activation(self.bottle_bn1(self.conv_bottle1(y4)))
        y6 = self.activation(self.bottle_bn2(self.conv_bottle2(y5)))
        y7 = self.activation(self.bottle_bn3(self.conv_bottle3(y6)))

        y = self.dense(self.flatten(y7))
        y = y.reshape(n_step, n_batch, -1)

        return y

'''class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 32,
        activation = nn.ELU
    ):
        super().__init__()

        self.out_dim = 256 #16896 #8960

        #for 820x410 images
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.dense1 = nn.Linear(16896, 512)
        self.dense2 = nn.Linear(512, self.out_dim)
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()

    def forward(self, x):
        #n_step, n_batch, n_channel, height, width = x.shape

        y1 = self.activation(self.bn1(self.conv1(x)))
        y2 = self.activation(self.bn2(self.conv2(y1)))
        y3 = self.activation(self.bn3(self.conv3(y2)))
        y4 = self.activation(self.bn4(self.conv4(y3)))
        y5 = self.activation(self.bn5(self.conv5(y4)))
        out = self.dense1(self.flatten(y5))
        out = self.dense2(out)
        #res1 = x.reshape(x.shape[0], -1)
        #res2 = y6.view(y6.shape[0], -1)
        #out = res1 + res2
        #out = self.activation(self.dense1(out))
        #out = self.dense2(out)

        return out'''