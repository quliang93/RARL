"""
Backbone: about Network (actor-critic framework) .
Designed by ZHAO Yuenan
Date: 2023-10-23
Backbone-I: ResNet-50
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from .utils import log_normal_density


def make_backbone(input_channels=10):
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size= 7, stride= 2, padding= 3, bias= False)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-2])
    global_avg_pooling = nn.AdaptiveAvgPool2d(1)
    fc_layer = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU()
    )
    resnet50_modified = nn.Sequential(
        resnet50,
        global_avg_pooling,
        nn.Flatten(),
        fc_layer
    )
    return resnet50_modified


def make_backbone_squeezenet(input_channels=10):
    """
    make backbones for CNNPolicy .
    :param input_channels: input channels, 10 = cad(1)+det(3)+dynamic(3)+route(3)
    :return: torch.tensor [1, 256]
    """
    squeezenet = models.squeezenet1_0()
    squeezenet.features[0] = nn.Conv2d(input_channels, 96, kernel_size=7, stride=2)
    squeezenet_features = squeezenet.features
    squeezenet_features = nn.Sequential(*list(squeezenet_features.children()))
    fc_layer = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU()
    )
    squeezenet_modified = nn.Sequential(
        squeezenet_features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        fc_layer
    )
    return squeezenet_modified


class CNNPolicy(nn.Module):
    """
    Agent Network.
    Backbone selected: SqueezeNet v1-0 .
    """
    def __init__(self, frames, action_space):
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))  # initial from zero .
        # self.logstd = nn.Parameter(torch.tensor([0.2, 0.2])) # we give them a initialized value, like 0.2
        # actor network
        # self.actor_bev_backbone = make_backbone(input_channels=frames)
        self.actor_bev_backbone = make_backbone_squeezenet(input_channels=frames)
        self.actor_fc = nn.Linear(256 + 2 + 2 + 2, 128) # proprioceptive obs dim = 6
        # self.actor1 = nn.Linear(128, 1)
        # self.actor2 = nn.Linear(128, 1)
        self.actor = nn.Linear(128, 2) # we map the mid-feature to 2-dim actions.

        # critic network
        self.critic_bev_backbone = make_backbone_squeezenet(input_channels=frames)
        self.critic_fc = nn.Linear(256 + 2 + 2 + 2, 128) # proprioceptive obs dim = 6
        self.critic = nn.Linear(128, 1)


    def forward(self, bev_obs, propri_obs):
        """
        Forward process .
        :param bev_obs: bev observation
        :param propri_obs: propri observation
        :return: value, action(sampled from Normal Distribution), logprob, mean (the predicted action from network)
        """
        a_bev = self.actor_bev_backbone(bev_obs)     # output: 256
        a = torch.cat((a_bev, propri_obs), dim= -1)  # output: 256 + 6
        a = F.relu(self.actor_fc(a))                 # output: 128
        # mean_steering = torch.tanh(self.actor1(a))   # output: 1
        # mean_throttle = torch.tanh(self.actor2(a))   # output: 1
        mean = torch.tanh(self.actor(a))
        # mean = torch.cat((mean_steering, mean_throttle), dim= -1)  # the true action inferred from Actor
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)             # action: sampled from Normal Distribution, [steering, throttle]
        logprob = log_normal_density(action, mean, std=std, log_std = logstd)

        # value
        v_bev = self.critic_bev_backbone(bev_obs)
        v = torch.cat((v_bev, propri_obs), dim= -1)
        v = F.relu(self.critic_fc(v))
        v = self.critic(v)

        return v, action, logprob, mean


    def evaluate_actions(self, bev_obs, propri_obs, action):
        """
        Evaluate actions
        :param bev_obs: bev observation
        :param propri_obs: propri observation
        :param action: action
        :return: value, logprob, dist_entropy(what's this ???)
        """
        v, _, _, mean = self.forward(bev_obs, propri_obs)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std= logstd, std= std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy



if __name__ == "__main__":
    n_frames = 10
    action_space = 2
    policy_network = CNNPolicy(n_frames, action_space)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # model = policy_network.to(device)
    policy_network.cuda()

    bev_tensor = torch.randn(1, 10, 128, 128)
    propri_tensor = torch.randn(1, 6)
    bev_tensor_gpu = bev_tensor.to(device)
    propri_tensor_gpu = propri_tensor.to(device)

    _, _, _, mean = policy_network(bev_tensor_gpu, propri_tensor_gpu)
    # print(f"bev feature shape: {mean[0]}, propri feature shape: {mean[1]}")
    print(mean[0])
    policy_network.evaluate_actions(bev_tensor_gpu, propri_tensor_gpu, mean[0])


    print(f"Evaluated !")