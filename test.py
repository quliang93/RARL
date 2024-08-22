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


def run(env: BEVENV, policy, action_bound: list):
    global_step = 0
    # collision_score = 1
    for id in range(0, length_of_test_episodes):
        collision_score = 1
        epside_passed = True
        lateral_off_total = 0
        candidate_collision_step = 0

        env.reset_test(scenarios_list[id])

        route_length = env.total_length # note the length of the current route length
        linear_vel = 0
        angular_vel = 0
        infraction_times= 0
        left_route_length = 0

        global_route_length = len(env.global_planner.original_global_waypoints)
        # prepare the adversary scenarios .
        # spawn_npc_subprocess = subprocess.Popen(["python", "Scenarios/adversary_scenarios_generator.py"])
        
        # Test BEVRL agent in normal crowded scenes, here we spawn pedestrian=60, vechiels=120
        spawn_npc_subprocess = subprocess.Popen(["python", "Scenarios/spawn_npc.py"])
        time.sleep(5)

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
            # env.run_step(scaled_action[0]) if env.get_traffic_light_state() == False else env.run_step([0, -1])
            env.render()
            # env.render_show(step)
            # env.render_velocity()
            # env.render_obs_action(scaled_action[0])
            r, terminal, result, current_dis_lateral, is_collision = env.test_reward(step)
            if result in ["invalid-stop", "over-speed", "collision", "run-off"]:
                epside_passed = False # that means the current test episode didn't pass successfully
                infraction_times += 1
                if infraction_times == 1:
                    left_route_length = len(env.global_planner.global_path_waypoints)


            if result == "collision":
                rospy.loginfo(f"Collision event notice !!!")
                collision_score = collision_score * 0.1

            # calculate the lateral off-lane distance of each step .
            lateral_off_total += current_dis_lateral


            ep_reward += r

            # get next state
            bev_obs_next = env.get_bev_img_obs()
            _, _, _, propri_obs_next = env.get_propriceptive_obs()
            # to calculate the average velocity.
            linear_vel += propri_obs_next[2]
            angular_vel += np.abs(propri_obs_next[3])

            state_next = [bev_obs_next, propri_obs_next]
            step += 1
            state = state_next

        # kill the adversary subprocess
        os.kill(spawn_npc_subprocess.pid, signal.SIGINT)
        time.sleep(2)

        episode_result = {"Scenarios_route": scenarios_list[id], "route_length": route_length,"pass": epside_passed,
                          "collision_score": collision_score,
                          "route_complete": (global_route_length - len(
                              env.global_planner.global_path_waypoints)) / global_route_length,
                          "safe_traveled_distance": global_route_length - left_route_length,
                          "average_speed": 3.6 * linear_vel / step,
                          "angular_speed": angular_vel / step,
                          "alod": lateral_off_total / step, "reward": ep_reward}
        print(episode_result)

        total_test_results.append(episode_result)

        for i in range(5):
            bev_env.run_step([0, 0])

    print(f"Recording All Test Results ...")
    with open(f"test_results_{json_name}.json", 'w', encoding= 'utf-8') as f:
        json.dump(total_test_results, f, ensure_ascii=False, indent=4)

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

    # 2. prepare the policy network and pretrained checkpoints ()
    policy_checkpoints_path = "checkpoints_I/best.pth"
    # policy_checkpoints_path = "./stage_I_best_5429.pth"
    # policy_checkpoints_path = "./5429_best.pth"
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

















