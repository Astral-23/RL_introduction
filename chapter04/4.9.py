import numpy as np
import matplotlib.pyplot as plt

def gambler_problem(ph : float):
    GOAL = 100
    GAMMA = 1
    value = np.zeros(GOAL + 1, dtype = np.float32)
    value[GOAL] = 0
    
    # 価値反復
    print("Value Iteration started.")
    while True:
        delta = -np.inf
        for s in range(1, GOAL):
            v = value[s]
            max_value = -np.inf
            max_a = None
            for a in range(1, s + 1):
                sum = 0
                sum += ph * (((s + a) >= GOAL) + GAMMA * value[min(s + a, GOAL)])
                sum += (1-ph) * (GAMMA * value[max(s - a, 0)])
                if max_value < sum:
                    max_value = sum
                    max_a = a
            value[s] = max_value
            delta = max(delta, abs(v - value[s]))
        print(f"delta = {delta:.6f}")
        if delta  < 1e-2:
            break
    print("Value Iteration finished.")
    
    # 方策復元
    policy = np.zeros(GOAL + 1, dtype = np.int32)
    for s in range(1, GOAL):
        max_value = -np.inf
        max_a = None
        for a in range(1, s + 1):
            sum = 0
            sum += ph * ((s + a) >= GOAL + GAMMA * value[min(s + a, GOAL)])
            sum += (1-ph) * (GAMMA * value[max(s - a, 0)])
            if max_value < sum:
                max_value = sum
                max_a = a
        policy[s] = max_a
    return value, policy

if __name__ == "__main__":

    phs = [0.1, 0.25, 0.5, 0.75, 0.9]
    fig, axes = plt.subplots(len(phs), 2, figsize=(len(phs)*2, len(phs)*3))
    for i, ph in enumerate(phs):
        vp = gambler_problem(ph)
        for j in range(2):
            axes[i][j].set_title(f"ph = {ph}")
            axes[i][j].plot(np.arange(0, 101), vp[j])
            axes[i][j].set_xlabel("Capital")
            axes[i][j].set_ylabel("Value estimates" if j == 0 else "Policy")
        
    plt.tight_layout()
    plt.show()
