import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

def bandit(k : int, first_value : int, step : int, epsilon : float, steady : bool, estimate_method : dict):
    '''
    k椀バンディット問題を1回シュミレーションする。
    Args:
        k (int): スロットの個数
        first_value (int): 最初の推定値。
        step (int): シミュレーションのステップ数。
        epsilon (float): ランダムなスロットを選ぶ確率
        steady (bool): 定常にするか否か
        estimate_method (dict): 推定価値の更新方法を指定する辞書。
            'name'キーで更新方法を指定し、必要に応じて追加のキーでパラメータを渡す。
            - 標本平均を使用する場合: {'name': 'ave'}
            - 固定ステップサイズを使用する場合: {'name': 'fix', 'alpha': float}
    Return:
        mse (np.ndarray, dtype=np.float32): 各ステップにおける推定値と真の価値の平均二乗誤差 (Mean Squared Error) を格納した配列。
        best_prob (np.ndarray, dtype=np.float32): 各ステップで最適なスロットを選択した確率を格納した配列。
    Note:
        スロットのvalue(期待値)は正規分布N(0, 1)に従う
        steady=Falseの場合、時間ステップ事にN(0, 0.01)に従う差分がvalueに追加される(スロットごとに独立)
    '''
    assert k >= 1, "k must be >= 1"
    assert step >= 1, "step must be >= 1"
    assert epsilon >= 0, "epsilon must be >= 0"
    assert epsilon <= 1, "epsilon must be <= 1"
    
    mse = np.zeros(step, dtype=np.float32)
    best_prob = np.zeros(step, dtype=np.float32)
    
    values = np.random.randn(k)
    estimated_values = np.full(k, first_value, dtype=np.float32)
    frequency =np.zeros(k)
    for t in range(step):
        # slotを選ぶ
        slot = None
        if np.random.random() < epsilon:
            slot = random.randint(0, k - 1)
        else:
            slot = np.argmax(estimated_values)
        frequency[slot] += 1
        reward = np.random.normal(values[slot], 1)
        
        # 推定価値を更新する
        if estimate_method['name'] == 'ave':
            estimated_values[slot] += (reward - estimated_values[slot]) / (frequency[slot])
        elif estimate_method['name'] == 'fix':
            estimated_values[slot] += (reward - estimated_values[slot]) * estimate_method['alpha']
        # ログを取る
        mse[t] += np.mean((estimated_values - values) ** 2)
        best_prob[t] += slot == np.argmax(values)
        
        # 非定常なら価値を更新
        if not steady:
            delta = np.random.normal(0, 0.01, k)
            values += delta
    return mse, best_prob


def bandit_many_times(k : int, first_value : int, step : int, epsilon : float, steady : bool, estimate_method : dict, experiment_num : int):
    '''
    k椀バンディット問題をexperiment_num回並列にシュミレーションする。
    '''
    core_num = os.cpu_count()
        
    with mp.Pool(processes=core_num) as pool:
        # pool.starmapを使って、引数をタプルで渡す
        results = pool.starmap(bandit, [(k, first_value, step, epsilon, steady, estimate_method)] * experiment_num)
        

    total_mse = np.zeros(step, dtype=np.float32)
    total_best_prob = np.zeros(step, dtype=np.float32)
    
    for mse, best_prob in results:
        total_mse += mse
        total_best_prob += best_prob
        
    average_mse = total_mse / experiment_num
    average_best_prob = total_best_prob / experiment_num
    return average_mse, average_best_prob
    
if __name__ == '__main__':
    k = 10
    first_value = 0
    step = 10000
    epsilon = 0.01
    experiment_num = 1000
    update = [
        {'name' : 'ave'},
        {'name' : 'fix', 'alpha' : 0.1}
    ]

    # グラフの設定
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10)) 
    for i in range(2):
        for j in range(2):
            update_method = update[j]['name']
            steady_bool = bool(i) 
            mse_avg, best_prob_avg = bandit_many_times(k, first_value, step, epsilon, steady_bool, update[j], experiment_num)
            axes[i, j].set_title(f"Steady={steady_bool}, epsilon={epsilon}, update = {update_method}")
            axes[i, j].plot(np.arange(1, step + 1), mse_avg, label='Average MSE')
            axes[i, j].plot(np.arange(1, step + 1), best_prob_avg, label='Average Best Action Probability')
            axes[i, j].set_xlabel('Step')
            axes[i, j].legend()
    plt.tight_layout() # サブプロット間のスペースを自動調整
    plt.show()