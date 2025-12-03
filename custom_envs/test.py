from dm_control.utils import rewards
import numpy as np
import matplotlib.pyplot as plt


def reward_function(input, vel=3):
    return rewards.tolerance(input, (vel, vel), margin=vel, sigmoid="linear", value_at_margin=0.0)


if __name__ == "__main__":
    # x [-1, 1]
    x = np.arange(-5, 5, 0.01)
    y = reward_function(x)
    plt.plot(x, y)
    plt.savefig("reward.png")
    plt.close()
