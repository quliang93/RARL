"""
Goal: A simple recovery train loop designed for STRA_RL project.
Date: 2023.12.30, By ZYN.
Notice:
    For using carla & carla-ros-bridge, the train-loop usually breakdown. So I prepare to
    add the checkpoint mechanism provided by PyTorch to make the Reward-Episode Curve look constant
    from scratch.
"""
import os
import time
import signal
import subprocess
import numpy as np
import rospy
import torch
import torch.nn as nn
from torch.optim import Adam
from Algorithm.ppo_modified import generate_action, transform_buffer
from Algorithm.ppo_modified import generate_train_data, ppo_update

from tensorboardX import SummaryWriter
from envs.bev_env_ours import BEVENV
from Algorithm.backbone import CNNPolicy
gpu_devices = torch.cuda.device_count()
print(f"***************************************************")
print(f"********* You have {gpu_devices} GPUs ! ***********")
print(f"***************************************************")
RL_TRAINING_DEVICE = gpu_devices - 1

writer = SummaryWriter("./stage_I_logs")

torch_seed = 5429
torch.manual_seed(torch_seed)

MAX_EPISODES = 8000
HORIZON = 1024
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 128
EPOCH = 4
COEFF_ENTROPY = 1e-3
CLIP_VALUE = 0.1
NUM_ENV = 1
ACT_SIZE = 2
BEV_INPUT_CHANNELS = 10
LEARNING_RATE = 5e-5


def run(env: BEVENV, policy, policy_path, action_bound: list, optimizer, start_episode = 0):
    buff = []
    global_update = 0
    global_step = 0
    every_update_average_reward = 0
    max_ep_r = -np.inf
    for id in range(start_episode, MAX_EPISODES):
        env.reset()
        spawn_npc_subprocess = subprocess.Popen(["python", "Scenarios/spawn_npc.py"])
        time.sleep(5)
        terminal = False
        ep_reward = 0
        step = 1

        # get initial state .
        bev_obs = env.get_bev_img_obs()
        _, _, _, propri_obs = env.get_propriceptive_obs()
        state = [bev_obs, propri_obs]
        while not terminal and not rospy.is_shutdown():
            state_list = []
            state_list.append(state)

            v, a, logprob, scaled_action = generate_action(env= env, state_list = state_list, policy= policy, action_bound= action_bound, device=RL_TRAINING_DEVICE)

            # do actions .
            env.run_step(scaled_action[0])

            env.render()
            r, terminal, result = env.reward_stcr_risk_aware(step)  # env.reward(step)
            ep_reward += r
            global_step += 1

            # get next state
            bev_obs_next = env.get_bev_img_obs()
            _, _, _, propri_obs_next = env.get_propriceptive_obs()
            state_next = [bev_obs_next, propri_obs_next]
            if global_step % HORIZON == 0:
                state_next_list = []
                state_next_list.append(state_next)
                last_v, _, _, _ = generate_action(env=env, state_list = state_next_list, policy= policy, action_bound=action_bound, device= RL_TRAINING_DEVICE)

            # add transitions in buff and update policy
            r_list = []
            r_list.append(r)
            terminal_list = []
            terminal_list.append(terminal)

            # env.index represents process number, process 0 means update network
            # 4. acquire next state .
            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                every_update_average_reward += np.mean(r_list)
                # print(f"current buff length: {len(buff)}")
                if len(buff) > HORIZON - 1:
                    ########################## update policy network ##########################
                    bev_batch, propri_batch, a_batch, r_batch, d_batch, l_batch, v_batch = transform_buffer(buff= buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma= GAMMA, values= v_batch,
                                                              last_value= last_v, dones= d_batch, lam=LAMDA)
                    memory = (bev_batch, propri_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)

                    rospy.loginfo(f"********** start training **********")
                    bev_env.run_step([0, 0]) # when start ppo training, do a brake action to make car stop .
                    ppo_update(policy= policy, optimizer= optimizer, batch_size= BATCH_SIZE, memory= memory,
                               epoch= EPOCH, coeff_entropy= COEFF_ENTROPY, clip_value= CLIP_VALUE, num_step= HORIZON,
                               num_env= NUM_ENV, propri_size= 6, bev_channels= BEV_INPUT_CHANNELS, bev_size=(128, 128),
                               act_size= ACT_SIZE, device= RL_TRAINING_DEVICE)  # the most important function in PPO algorithm .
                    ###########################################################################
                    rospy.loginfo(f"********** end ppo update **********")
                    buff = []
                    global_update += 1
                    writer.add_scalar("global_update_reward", every_update_average_reward, global_update)
                    every_update_average_reward = 0

            step += 1
            state = state_next

        for i in range(5):
            bev_env.run_step([0, 0])

        os.kill(spawn_npc_subprocess.pid, signal.SIGINT)
        time.sleep(2)

        # torch save, for recovery mode training, we save last.pth, best.pth and every 5 interval episodes.
        if env.index == 0:
            # torch save the best.pth
            if ep_reward > max_ep_r:
                max_ep_r = ep_reward
                # torch.save(policy.state_dict(), policy_path + f"/best.pth")
                torch.save({
                    "episode_id": id,
                    "episode_reward": ep_reward,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": policy.state_dict()
                }, policy_path + "best.pth")

            # torch save the last.pth
            torch.save({
                "episode_id": id,
                "episode_reward": ep_reward,
                "optimizer": optimizer.state_dict(),
                "state_dict": policy.state_dict()
            }, policy_path + "last.pth")

            # torch save after each 5 interval
            name = f"TOWN02_OURS_STAGEI_{id}_{ep_reward}"
            if id % 10 == 0:
                torch.save({
                    "episode_id": id,
                    "episode_reward": ep_reward,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": policy.state_dict()
                }, policy_path + f"{name}.pth")


            writer.add_scalar("episode reward", ep_reward, id)
            print('Env %02d, Episode %05d, setp %03d, global_steps %d, Reward %-5.4f,  %s' % \
                  (env.index, id, step, global_update, ep_reward, result))

    env.reset()  # if all train episodes finished, env reset()


if __name__ == "__main__":
    # 1. initializing environment .
    bev_range = [-10, -20, 30, 20]
    bev_img_size = [512, 512]
    ego_vehicle_dimensions = [2.52, 1.47]
    target_velocity = 5  # from 5 -> 10 (max speed)
    bev_env = BEVENV(bev_range, bev_img_size, ego_vehicle_dimensions, target_velocity)
    reward = None
    action_bound = [[-1, -1], [1, 1]]  # action boundary, [steering, throttle]

    # 2. prepare the policy network and pretrained checkpoints (yes or not)
    policy = CNNPolicy(frames= BEV_INPUT_CHANNELS, action_space= 2)
    policy.cuda(device= RL_TRAINING_DEVICE)
    opt = Adam(policy.parameters(), lr= LEARNING_RATE)
    policy_checkpoints_pth = "checkpoints_I/"
    last_epoch_checkpoint = policy_checkpoints_pth + "last.pth"

    if os.path.exists(last_epoch_checkpoint):
        print("##################################################")
        print("################ Recovery from Last ##############")
        print("##################################################")
        # 1. torch load the latest epoch's checkpoint
        resume_checkpoint = torch.load(last_epoch_checkpoint)
        start_episode = resume_checkpoint["episode_id"]
        state_dict = resume_checkpoint["state_dict"]
        policy.load_state_dict(state_dict)
        policy.cuda(device=RL_TRAINING_DEVICE)
        opt.load_state_dict(resume_checkpoint["optimizer"])

    else:
        print("##################################################")
        print("############## Train from scratch ! ##############")
        print("##################################################")
        start_episode = 0


    # 3. start training .
    try:
        run(env=bev_env, policy=policy, policy_path=policy_checkpoints_pth,
            action_bound=action_bound, optimizer=opt, start_episode= start_episode)

    except KeyboardInterrupt:
        pass
