# Date: 2024-3-27
# By: ZYN
# Notice: This script is designed for calculating different metrics for Ablation Study.

"""
Readme: The whole test process is to evaluate our method with different methods or ablation settings.
So, the Metric we select is the four items:

Success Rate (SR): if current episode is accomplished without any collision, curb invasion, and navigate to target point
finally, we regard this episode as ONE successful episode.

Collision Rate (CR): we calculate the candidate collision steps ratio to the all episode steps. Once the distance between
self-vehicle to nearest obstacle(vehicle, pedestrian) is less than a Threshold, while its velocity is not zero. We regard
the step is candidate collision step.

Average Lane-Off Distance (ALOD): average of distance between self-vehicle to lane center of each step.

Average Reward (AR): average reward of test episodes.

"""
import json
import os
import signal
import subprocess
import time
import random
import rospy
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from Algorithm.ppo_modified import generate_action_no_sampling
from tensorboardX import SummaryWriter
from envs.bev_env_ours import BEVENV, scenarios_route
from Algorithm.backbone import CNNPolicy

writer = SummaryWriter("./ablation_study_writer")

BEV_INPUT_CHANNELS = 10

scenarios_list = list(scenarios_route.keys())
length_of_test_episodes = len(list(scenarios_route.keys()))
json_name = list(scenarios_route.keys())[0].split("_")[0]


total_test_results = []

adversarial_scenarios = ["crossing"] # stop, crossing, merge, reverse


def run(env: BEVENV, policy, action_bound: list):

    for id in range(length_of_test_episodes):

        env.reset_test(scenarios_list[id])
        # prepare the adversary scenarios .
        scenarios_type = random.choice(adversarial_scenarios)
        spawn_npc_subprocess = subprocess.Popen(["python", "Scenarios/adversary_scenarios_generator_v3.py", scenarios_type, scenarios_list[id]])
        time.sleep(1)

        terminal = False
        result = None
        ep_reward = 0
        step = 1
        bev_obs = env.get_bev_img_obs()
        _, _, _, propri_obs = env.get_propriceptive_obs()
        state = [bev_obs, propri_obs]
        while not terminal and not rospy.is_shutdown():
            state_list = []
            state_list.append(state)
            mean, scaled_action = generate_action_no_sampling(env=env, state_list= state_list, policy = policy, action_bound= action_bound)

            # do actions .
            env.run_step(scaled_action[0])
            # env.render()
            env.render_show(step)
            r, terminal, result, current_dis_lateral, is_collision = env.test_reward_adversarial(step)
            if result in ["invalid-stop", "over-speed", "collision", "run-off"]:
                epside_passed = False # that means the current test episode didn't pass successfully

            ep_reward += r

            # get next state
            bev_obs_next = env.get_bev_img_obs()
            _, _, _, propri_obs_next = env.get_propriceptive_obs()
            state_next = [bev_obs_next, propri_obs_next]
            step += 1
            state = state_next

        # kill the adversary subprocess
        os.kill(spawn_npc_subprocess.pid, signal.SIGINT)
        time.sleep(2)

        for i in range(5):
            bev_env.run_step([0, 0])

    print(f"Quit !")


if __name__ == "__main__":
    # 1. initializing environment
    bev_range = [-10, -20, 30, 20]
    bev_img_size = [512, 512]
    ego_vehicle_dimensions = [2.52, 1.47]
    target_velocity = 5
    bev_env = BEVENV(bev_range, bev_img_size, ego_vehicle_dimensions, target_velocity)

    reward = None
    action_bound = [[-1, -1], [1, 1]]

    # 2. prepare the policy network and pretrained checkpoints
    policy_checkpoints_path = "checkpoints_II/best.pth"
    policy = CNNPolicy(frames= BEV_INPUT_CHANNELS, action_space = 2)
    policy.cuda()
    policy.eval() # eval mode .

    resume_checkpoint = torch.load(policy_checkpoints_path)
    state_dict = resume_checkpoint["state_dict"]
    policy.load_state_dict(state_dict)

    print(f"*************** Test Policy Prepared ! ***************")

    # 3. start testing ...
    try:
        run(bev_env, policy, action_bound)
    except KeyboardInterrupt:
        pass
