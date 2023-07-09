import numpy as np
import math
import pandas as pd
from statistics import mean, stdev
from tqdm.notebook import trange, tqdm
import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append("../src")

from main import Agent, run
# from make_sfm import run_sfm
from social_force_model import SocialForceModel
from destination_choice_model import DestinationChoiceModel



# Agent
class AgentSFM():
    def __init__(self, no, t0, r, v0, loc, vel, dest):
        self.no = no
        self.t0 = t0
        self.r = r
        self.v0 = v0
        self.loc = loc
        self.vel = vel
        self.dest = dest
        self.done = False

def run_SFM(trajectories, mean_v, mean_vx, mean_vy, num_step, sfm, dests):
     # initialize agents
    num_agent = len(trajectories)
    agents = {}
    for i in range(num_agent):
        agents[i] = AgentSFM(no=i,
                          t0=trajectories[i][0][-1],
                          r=0.2,
                          v0=np.array(mean_v[i]),
                          loc=[np.array([np.nan, np.nan, np.nan])],
                          vel=[np.array([np.nan, np.nan])],
                          dest=[np.array([np.nan, np.nan])],
                          )

    t = 0
    while t < num_step:  # until all pedestrians move to right
        for i in range(num_agent):
            # generate agents
            if agents[i].t0 == t:
                agents[i].loc[t] = np.array(
                    [trajectories[i][0][0], trajectories[i][0][1], trajectories[i][0][2]])
                agents[i].vel[t] = np.array([np.mean(mean_vx), np.mean(mean_vy)])
                agents[i].dest[t] = dests[i][:2]
                # print("agents[i].dest[t]", agents[i].dest[t])
                print(f"agents[{i}].dest[{t}]", agents[i].dest[t])


            # real
            # if (agents[i].type == "real"):
            #     agents[i].loc.append(
            #         np.array([real_trajectories[i].x[t], real_trajectories[i].y[t]]))
            #     agents[i].vel.append(
            #         np.array([real_trajectories[i].vel_x[t], real_trajectories[i].vel_y[t]]))
            #     if real_trajectories[i].choice[t] - 1 in [0, 1]:
            #         agents[i].dest.append(
            #             dests[int(real_trajectories[i].choice[t]-1)])
            #     else:
            #         agents[i].dest.append(np.array([np.nan, np.nan]))

            # model
            if (agents[i].t0 <= t) and (agents[i].done == False):

                dest = agents[i].dest[t]

                loc_next, vel_next = sfm.step(t, i, agents, dest)

                agents[i].loc.append(loc_next)
                agents[i].vel.append(vel_next)
                agents[i].dest.append(dest)
                # print(loc_next)
                # print(vel_next)
                # print(dest)

                if np.linalg.norm(loc_next[:2] - dest) <= 0.1:
                    agents[i].done = True
            else:
                agents[i].loc.append(np.array([np.nan, np.nan, np.nan]))
                agents[i].vel.append(np.array([np.nan, np.nan]))
                agents[i].dest.append(np.array([np.nan, np.nan]))
        t += 1

    return agents

def read_trajectory(agents):
    print("read trajectory")
    print(agents)
    trajectories = []
    for i in range(len(agents)):
        traj = agents[i].loc
        traj = [row.tolist() for row in traj if not np.isnan(row).any()]
        traj = convert_last_to_int(traj)
        trajectories.append(traj)
    
    destinations = []
    for ag in trajectories:
        dest = ag[-1]
        destinations.append(dest)
    
    return trajectories, destinations

def convert_to_tuples(lst):
    return [tuple(item) for item in lst]

def convert_to_tuples2(lst):
    return [[tuple(item)] for item in lst]

def convert_last_to_int(lst):
    return [(item[0], item[1], int(item[2])) for item in lst]



# __all__ = ['runSFM']

if __name__ == "__main__":
    data_path = '/Users/juneaist/work/equivariant-PIML/data/GC_dataset/GC_Dataset_ped1-12685_time2344-2404_interp9_xrange5-25_yrange15-35'
    # data_path = '/Users/juneaist/work/equivariant-PIML/data/GC_dataset/GC_Dataset_toy1'
    load_path = data_path + '.npy'
    data = np.load(load_path, allow_pickle=True)
    meta_data, trajectories, destinations, obstacles = data

    mean_v = []
    for u in trajectories:
        first = u[0][:2]
        last = u[-1][:2]
        tau = u[-1][-1] - u[0][-1]
        first, last = np.array(first), np.array(last)
        v = (last - first) / tau
        v = v.tolist()
        mean_v.append(v)
    mean_vx = [v[0] for v in mean_v]
    mean_vy = [v[1] for v in mean_v]
    
    params_sfm = {
        "dt": 1/30,
        "A1": 2.1,
        "B": 0.3,
        "A2": 1,
        "tau": 0.5,
        "phi": 100,
        "c": 0.5
    }
    walls = np.array(obstacles)
    walls_points = walls.tolist()
    sfm = SocialForceModel(params_sfm, walls_points)
    num_steps = max([u[-1][-1] for u in trajectories]) + 1
    # num_steps = 100 # for test
    # dests = [[x[0] for x in sublist] for sublist in destinations]
    dests = [list(item[0]) for item in destinations]
    print(dests)

    agents = run_SFM(trajectories, mean_v, mean_vx, mean_vy, num_steps, sfm, dests)
    # print(agents[0].loc)
    
    trajectories, destinations = read_trajectory(agents)

    converted_trajectories = []
    for traj in trajectories:
        converted_traj = convert_to_tuples(traj)
        converted_trajectories.append(converted_traj)
    trajectories = converted_trajectories
    destinations = convert_to_tuples2(destinations)
    print("destinations")
    print(destinations)

    save_path = data_path + '_SFM.npy'
    data = np.array((meta_data, trajectories, destinations, obstacles), dtype=object)
    np.save(save_path, data)
    print("saved for ", save_path)