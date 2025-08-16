import numpy as np
from scipy.stats import poisson
import time

def jack_rental_car_problem():
    MAX_CARS_STORED = 20
    MAX_CARS_MOVED = 5
    MAX_CARS_RETURNED = MAX_CARS_STORED + MAX_CARS_MOVED
    CAR_MOVING_COST = 2
    CAR_LEND_REWARD = 10
    MAX_REWARD  = MAX_CARS_STORED * CAR_LEND_REWARD * 2 
    GAMMA = 0.9
    LAMBDA_IN = [3, 4]
    LAMBDA_OUT = [3, 2]
    
    # valueとpolicyを初期化。policyは決定的方策
    value = np.zeros((MAX_CARS_STORED + 1, MAX_CARS_STORED + 1), dtype=np.float32)
    policy = np.zeros((MAX_CARS_STORED + 1, MAX_CARS_STORED + 1), dtype=np.int32)
    
    poisson_table = np.zeros((MAX_CARS_RETURNED + 1, MAX_CARS_RETURNED+ 1))
    for i in range(MAX_CARS_RETURNED + 1):
        for j in range(MAX_CARS_RETURNED + 1):
            poisson_table[i][j] = poisson.pmf(i, j)
    
    # Pr[r + MAX_REWARD][i][j][a + MAX_CARS_MOVED] := p(r  | (i, j), a)
    Pr = np.zeros((MAX_REWARD * 2 + 1, MAX_CARS_STORED + 1, MAX_CARS_STORED + 1, MAX_CARS_MOVED * 2 + 1))
    # Ps[i'][j'][i][j][a + MAX_CARS_MOVED] := p((i', j') | (i, j), a)
    Ps = np.zeros((MAX_CARS_STORED + 1, MAX_CARS_STORED + 1, MAX_CARS_STORED + 1,MAX_CARS_STORED + 1, MAX_CARS_MOVED * 2 + 1 ))
    
    # p(r | (i, j), a) を計算
    # 探索対象は 貸し出す車の数
    for i in range(MAX_CARS_STORED + 1):
        for j in range(MAX_CARS_STORED + 1):
            for a in range(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1):
                car_i = i - a
                car_j = j + a
                car_i = min(car_i, MAX_CARS_STORED)
                car_j = min(car_j, MAX_CARS_STORED)
                if car_i < 0 or car_j < 0:
                    continue
                for lent_i in range(car_i + 1):
                    for lent_j in range(car_j + 1):
                        reward = (lent_i + lent_j) * CAR_LEND_REWARD - abs(a) * CAR_MOVING_COST
                        proba = poisson_table[lent_i][LAMBDA_OUT[0]] * poisson_table[lent_j][LAMBDA_OUT[1]]
                        Pr[reward + MAX_REWARD][i][j][a + MAX_CARS_MOVED] += proba
        
    # p((i', j') | (i, j), a) を計算
    # p((i', j') | (i, j), a) = p(i' | i, a) * p(j' | j, a) 
    # p(i' | i, a)を先に計算する     
    # 探索対象は 返ってくる車の数、貸す車の数               
    P = np.zeros((2, MAX_CARS_STORED + 1, MAX_CARS_STORED + 1, MAX_CARS_MOVED * 2 + 1))
    for type in range(2):
        for i in range(MAX_CARS_STORED + 1):
            for a in range(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1):
                car = i
                if type == 0:
                    car -= a
                else:
                    car += a
                car = min(car, MAX_CARS_STORED)
                if car < 0:
                    continue
                for lent in range(car + 1):
                    for returned in range(MAX_CARS_RETURNED + 1):
                            ni = car - lent + returned
                            ni = min(ni, MAX_CARS_STORED)
                            if 0 <= ni:
                                P[type][ni][i][a + MAX_CARS_MOVED] += poisson_table[lent][LAMBDA_OUT[type]] * poisson_table[returned][LAMBDA_IN[type]]
                                
                                
    # p((i', j') | (i, j), a) を計算                        
    for i in range(MAX_CARS_STORED + 1):
        for j in range(MAX_CARS_STORED + 1):
            for ni in range(MAX_CARS_STORED + 1):
                for nj in range(MAX_CARS_STORED + 1):
                    for a in range(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1):
                        Ps[ni][nj][i][j][a + MAX_CARS_MOVED] = P[0][ni][i][a + MAX_CARS_MOVED] * P[1][nj][j][a + MAX_CARS_MOVED]

    loop_counter = 0
    while True:
        loop_counter += 1
        print(f"Loop {loop_counter} started.")
        # 方策評価
        print("Policy Evaluation started.")
        while True:
            delta= -np.inf
            for i in range(MAX_CARS_STORED + 1):
                for j in range(MAX_CARS_STORED + 1):
                    v = value[i][j]
                    sum = 0
                    for r in range(-MAX_REWARD, MAX_REWARD + 1):
                        sum += Pr[r + MAX_REWARD][i][j][policy[i][j] + MAX_CARS_MOVED] * r
                    for ni in range(MAX_CARS_STORED + 1):
                        for nj in range(MAX_CARS_STORED +1):
                            sum += Ps[ni][nj][i][j][policy[i][j] + MAX_CARS_MOVED] * GAMMA * value[ni][nj]
                    value[i][j] = sum
                    delta = max(delta, abs(v - value[i][j]))
            print(f"    Policy Evaluation: delta = {delta:.6f}")
            if delta < 1e-4:
                print("Policy Evaluation finished.")
                break   
            
        # 方策改善
        print("Policy Improvement started.")
        policy_stable = True
        for i in range(MAX_CARS_STORED + 1):
            for j in range(MAX_CARS_STORED + 1):
                max_value = -np.inf
                max_a = None
                for a in range(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1):
                    sum = 0
                    for r in range(-MAX_REWARD, MAX_REWARD + 1):
                        sum += Pr[r + MAX_REWARD][i][j][a + MAX_CARS_MOVED] * r
                    for ni in range(MAX_CARS_STORED + 1):
                        for nj in range(MAX_CARS_STORED +1):
                            sum += Ps[ni][nj][i][j][a + MAX_CARS_MOVED] * GAMMA * value[ni][nj]
                    if max_value < sum:
                        max_value = sum
                        max_a = a
                if policy[i][j] != max_a:
                    policy_stable = False
                policy[i][j] = max_a
        if policy_stable:
            break
        print("Policy Improvement finished.")
        print(f"loop {loop_counter} finished.")
    print("All loops finished.")
    return value, policy

if __name__ == '__main__':
    start_time = time.time()
    value, policy = jack_rental_car_problem()
    end_time = time.time()
    
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    
    # 結果の可視化（例）
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 方策のヒートマップ
    sns.heatmap(policy, ax=axes[0], cmap="viridis")
    axes[0].set_title("Optimal Policy (Number of cars moved from loc 1 to 2)")
    axes[0].set_xlabel("Cars at Location 2")
    axes[0].set_ylabel("Cars at Location 1")
    axes[0].invert_yaxis()

    # 価値関数のヒートマップ
    sns.heatmap(value, ax=axes[1], cmap="plasma")
    axes[1].set_title("Optimal Value Function")
    axes[1].set_xlabel("Cars at Location 2")
    axes[1].set_ylabel("Cars at Location 1")
    axes[1].invert_yaxis()

    
    plt.tight_layout()
    plt.show()