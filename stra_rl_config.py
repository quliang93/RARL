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

import os
import signal
import subprocess
import time
import rospy
from tensorboardX import SummaryWriter
from envs.bev_env_ours import BEVENV


writer = SummaryWriter("./ablation_study_writer")

BEV_INPUT_CHANNELS = 10


def run(env: BEVENV):
    for id in range(0, 100):
        bev_env.reset()
        spawn_npc_subprocess = subprocess.Popen(["python", "Scenarios/spawn_npc.py"])
        time.sleep(5)

        terminal = False
        result = None

        step = 1
        bev_obs = env.get_bev_img_obs()
        _, _, _, propri_obs = env.get_propriceptive_obs()
        state = [bev_obs, propri_obs]
        while not terminal and not rospy.is_shutdown():
            state_list = []
            state_list.append(state)
            env.run_step([0, 1])
            env.render()
            r, terminal, result, current_dis_lateral, is_collision = env.test_reward(step)

        rospy.loginfo(f"Episode result: {result}")

        os.kill(spawn_npc_subprocess.pid, signal.SIGINT)
        time.sleep(2)

        for i in range(5):
            bev_env.run_step([0, 0])

    print(f"Quit !")


if __name__ == "__main__":
    bev_range = [-10, -20, 30, 20]
    bev_img_size = [512, 512]
    ego_vehicle_dimensions = [2.52, 1.47]
    target_velocity = 5
    bev_env = BEVENV(bev_range, bev_img_size, ego_vehicle_dimensions, target_velocity)
    reward = None
    action_bound = [[-1, -1], [1, 1]]
    print(f"*************** Test Environment Prepared ! ***************")

    # 3. start testing ...
    try:
        run(bev_env)
    except KeyboardInterrupt:
        pass