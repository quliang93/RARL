import os
import torch
import socket
import logging
import numpy as np
from torch.nn import functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


def transform_buffer(buff):
    bev_batch, propri_batch, a_batch, r_batch, d_batch, l_batch, v_batch = [], [], [], [], [], [], []

    bev_temp, propri_temp = [], []

    for e in buff:
        for state in e[0]:
            bev_temp.append(state[0])
            propri_temp.append(state[1])

        bev_batch.append(bev_temp)
        propri_batch.append(propri_temp)

        bev_temp = []
        propri_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    bev_batch = np.asarray(bev_batch)
    propri_batch = np.asarray(propri_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return bev_batch, propri_batch, a_batch, r_batch, d_batch, l_batch, v_batch


def generate_action(env, state_list, policy, action_bound, device=0):
    """
    :param env: BEVENV
    :param state_list: [bev_obs, propri_obs]
    :param policy: policy network
    :param action_bound:
    :param device = 0 , which GPU device you choose as training device .
    :return:
            v: value ;
            a: action from normal distribution ;
            logprob: log probability ;
            scaled_action: action clipped according action bound ;
    """
    if env.index == 0:
        bev_list, propri_list = [], []
        for i in state_list:
            bev_list.append(i[0])
            propri_list.append(i[1])

        bev_list = np.asarray(bev_list)
        propri_list = np.asarray(propri_list)

        bev_list = torch.from_numpy(bev_list).float().cuda(device) # [1, 10, 224, 224]
        # print(bev_list.size())
        propri_list = torch.from_numpy(propri_list).float().cuda(device) # [1, 6]
        # print(propri_list.size())
        # print(f"bev_list tensor size: {bev_list.size()}")
        # print(f"propri_list tensor size: {propri_list.size()}")

        v, a, logprob, mean = policy(bev_list, propri_list)
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min= action_bound[0], a_max= action_bound[1])

    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action


def generate_action_no_sampling(env, state_list, policy, action_bound, device = 0):
    if env.index == 0:
        bev_list, propri_list = [], []
        for i in state_list:
            bev_list.append(i[0])
            propri_list.append(i[1])

        bev_list = np.asarray(bev_list)
        propri_list = np.asarray(propri_list)

        bev_list = torch.from_numpy(bev_list).float().cuda(device)  # [1, 10, 224, 224]

        propri_list = torch.from_numpy(propri_list).float().cuda(device)  # [1, 6]

        _, _, _, mean = policy(bev_list, propri_list)
        mean = mean.data.cpu().numpy()
        scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])

    else:
        mean = None
        scaled_action = None

    return mean, scaled_action


def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        targets[t, :] = gae + values[t, :]

    advs = targets - values[:-1, :]
    return targets, advs


def ppo_update(policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2, num_step=2048,
               num_env=1, propri_size=6, bev_channels=10, bev_size=(224, 224), act_size=2, device = 0):
    bevs, propris, actions, logprobs, targets, values, rewards, advs = memory
    advs = (advs - advs.mean()) / advs.std()

    bevs = bevs.reshape((num_step*num_env, bev_channels, bev_size[0], bev_size[1]))
    # print(f"bevs shape: {bevs.shape}")
    propris = propris.reshape((num_step*num_env, propri_size))
    # print(f"propris shape: {propris.shape}")
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size= batch_size,
                               drop_last= False)
        for i, index in enumerate(sampler):
            sampled_bevs = torch.from_numpy(bevs[index]).float().cuda(device)
            sampled_propris = torch.from_numpy(propris[index]).float().cuda(device)

            sampled_actions = torch.from_numpy(actions[index]).float().cuda(device)
            sampled_logprobs = torch.from_numpy(logprobs[index]).float().cuda(device)
            sampled_targets = torch.from_numpy(targets[index]).float().cuda(device)
            sampled_advs = torch.from_numpy(advs[index]).float().cuda(device)
            # print(f"sampled bevs size: {sampled_bevs.size()}")
            # print(f"sampled propris size: {sampled_propris.size()}")
            # print(f"sampled actions size: {sampled_actions.size()}")
            # print(f"sampled logprobs size: {sampled_logprobs.size()}")
            # print(f"sampled targets size: {sampled_targets.size()}")
            # print(f"sampled advs size: {sampled_advs.size()}")
            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_bevs, sampled_propris, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            # loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            loss = policy_loss + value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), \
                                                     float(dist_entropy.detach().cpu().numpy())
            logger_ppo.info(f"{info_p_loss}, {info_v_loss}, {info_entropy}")

    print("update")