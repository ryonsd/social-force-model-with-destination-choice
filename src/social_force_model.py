import numpy as np

# Operational level

class SocialForceModel:

    def __init__(self, params, environment):
        self.dt = params["dt"]
        self.A1 = params["A1"]
        self.B = params["B"]
        self.A2 = params["A2"]
        self.tau = params["tau"]
        self.phi = params["phi"]
        self.c = params["c"]

        self.environment = environment

    def social_force(self, t, i, agents, destination, delta=1e-3):

        # 1. to destination: F_id
        e_it = (destination - agents[i].loc[t]) / \
            np.linalg.norm(destination - agents[i].loc[t])
        F_id = (agents[i].v0 * e_it - agents[i].vel[t]) / self.tau

        # 2. from other pedestrians: F_ij
        F_ij = np.array([0, 0])
        for j in range(len(agents)):
            f_ij = np.array([0, 0])
            if 1 * np.isnan(agents[j].loc[t][0]) == 0:
                if j != i:
                    distance = np.linalg.norm(
                        agents[i].loc[t] - agents[j].loc[t])
                    n_ij = (agents[j].loc[t] - agents[i].loc[t]) / distance
                    f_ij = -1 * self.A1 * np.exp(-distance / self.B) * n_ij

                    e_jd = (agents[j].dest[t] - agents[j].loc[t]) / \
                        np.linalg.norm(agents[j].dest[t] - agents[j].loc[t])
                    if np.dot(f_ij, e_jd) >= np.linalg.norm(f_ij) * np.cos(np.angle(self.phi)):
                        w = 1
                    else:
                        w = self.c
                    f_ij = w * f_ij
            F_ij = F_ij + f_ij

        # 3. from objects: F_iW
        F_iW = np.array([0, 0])
        for w in self.environment:
            distance = np.linalg.norm(agents[i].loc[t] - np.array(w))
            f_iW = 0
            n_iW = (np.array(w) - agents[i].loc[t]) / distance
            f_iW = -1 * self.A2 * np.exp(-distance / agents[i].r) * n_iW
            F_iW = F_iW + f_iW
        #
        F = F_id + F_ij + F_iW
        return F

    def capped_velocity(self, desired_velocity, v_max):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        if desired_speeds <= v_max:
            g = 1
        else:
            g = v_max / desired_speeds
        return desired_velocity * g

    def step(self, t, i, agents, destination):
        loc = agents[i].loc[t]
        vel = agents[i].vel[t]
        F = self.social_force(t, i, agents, destination)
        v_next = vel + F * self.dt
        v_next = self.capped_velocity(v_next, agents[i].v0*1.3)
        loc_next = loc + v_next * self.dt

        return loc_next, v_next
