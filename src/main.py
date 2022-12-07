import numpy as np

# Agent
class Agent():
    def __init__(self, no, t0, r, v0, loc, vel, dest, type_):
        self.no = no
        self.t0 = t0
        self.r = r
        self.v0 = v0
        self.loc = loc
        self.vel = vel
        self.dest = dest
        self.type = type_
        self.done = False


def run(ped_info, num_step, sfm, dcm, dests, real_trajectories=None):

    # initialize agents
    num_agent = len(ped_info)
    agents = {}
    for i in range(num_agent):
        agents[i] = Agent(no=ped_info.ID[i],
                          t0=ped_info.frame_s[i],
                          r=0.2,
                          v0=ped_info.speed_mean[i],
                          loc=[np.array([np.nan, np.nan])],
                          vel=[np.array([np.nan, np.nan])],
                          dest=[np.array([np.nan, np.nan])],
                          type_=ped_info.type[i])

    t = 0
    while t < num_step:  # until all pedestrians move to right
        for i in range(num_agent):
            # generate agents
            if agents[i].t0 == t:
                agents[i].loc[t] = np.array(
                    [ped_info.start_x[i], ped_info.start_y[i]])
                agents[i].vel[t] = np.array([ped_info.speed_mean.mean(), 0])
                agents[i].dest[t] = dests[0]

            # real
            if (agents[i].type == "real"):
                agents[i].loc.append(
                    np.array([real_trajectories[i].x[t], real_trajectories[i].y[t]]))
                agents[i].vel.append(
                    np.array([real_trajectories[i].vel_x[t], real_trajectories[i].vel_y[t]]))
                if real_trajectories[i].choice[t] - 1 in [0, 1]:
                    agents[i].dest.append(
                        dests[int(real_trajectories[i].choice[t]-1)])
                else:
                    agents[i].dest.append(np.array([np.nan, np.nan]))

            # model
            elif (agents[i].t0 <= t) and (agents[i].done == False):

                if t % 15 == 0:
                    dest = dcm.choice(t, i, agents)
                else:
                    dest = agents[i].dest[t]

                loc_next, vel_next = sfm.step(t, i, agents, dest)

                agents[i].loc.append(loc_next)
                agents[i].vel.append(vel_next)
                agents[i].dest.append(dest)

                if np.linalg.norm(loc_next - dest) <= 0.2:
                    agents[i].done = True
            else:
                agents[i].loc.append(np.array([np.nan, np.nan]))
                agents[i].vel.append(np.array([np.nan, np.nan]))
                agents[i].dest.append(np.array([np.nan, np.nan]))
        t += 1

    return agents


if __name__ == "__main__":
    agent = Agent()
