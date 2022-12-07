import numpy as np

# Tactical level
class DestinationChoiceModel:

    def __init__(self, params, destinations):
        self.params = params
        self.dests = destinations

    def calc_theta(self, a, b):
        ab = np.dot(a, b)
        a_len = np.linalg.norm(a)
        b_len = np.linalg.norm(b)
        theta = np.arccos(ab / (a_len * b_len))
        return np.rad2deg(theta)

    def choice(self, t, i, agents):
        # calculation of variables related to route choice

        # distance
        dist_1 = np.linalg.norm(agents[i].loc[t] - self.dests[0])
        dist_2 = np.linalg.norm(agents[i].loc[t] - self.dests[1])

        # previous choice
        pre_choice_1 = 1 * (sum(agents[i].dest[t] == self.dests[0]) == 2)
        pre_choice_2 = 1 * (sum(agents[i].dest[t] == self.dests[1]) == 2)

        # others choice
        oth_choice_front_1 = 0
        oth_choice_front_2 = 0
        oth_choice_behind_1 = 0
        oth_choice_behind_2 = 0
        for j in range(len(agents)):
            if j != i and ((agents[j].t0 <= t) and (agents[j].done == False)):
                deg = self.calc_theta(
                    agents[i].vel[t], (agents[j].loc[t] - agents[i].loc[t]))
                if 1 * sum(agents[j].dest[t] == self.dests[0]) == 2:
                    if deg <= 100:
                        oth_choice_front_1 += 1
                    else:
                        oth_choice_behind_1 += 1
                else:
                    if deg <= 100:
                        oth_choice_front_2 += 1
                    else:
                        oth_choice_behind_2 += 1

        # calculation of utility
        V_dest1 = self.params["distance"] * dist_1 + self.params["previous_choice"] * pre_choice_1 + \
            self.params["others_choice_front"] * oth_choice_front_1 + \
            self.params["others_choice_behind"] * oth_choice_behind_1
        V_dest2 = self.params["distance"] * dist_2 + self.params["previous_choice"] * pre_choice_2 + \
            self.params["others_choice_front"] * oth_choice_front_2 + \
            self.params["others_choice_behind"] * \
            oth_choice_behind_2 + self.params["ASC_2"]

        # choice probability
        p_dest1 = np.exp(V_dest1) / (np.exp(V_dest1) + np.exp(V_dest2))
        p = [p_dest1, 1 - p_dest1]

        # choice
        dest = self.dests[int(np.random.choice([0, 1], p=p))]

        return dest
