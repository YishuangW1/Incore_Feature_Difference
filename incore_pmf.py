import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad
import json
import os
import csv  # 引入 CSV 模块

# --- 参数定义 ---
archetype_params = {
    't6': {
        'meanlog': np.log([3.415, 3.625, 3.895, 4.075]),
        'sdlog': np.array([0.11, 0.11, 0.11, 0.21])
    },
    't13': {
        'meanlog': np.log([3.255, 3.655, 3.875, 4.145]),
        'sdlog': np.array([0.12, 0.11, 0.11, 0.17])
    },
    't19': {
        'meanlog': np.log([3.425, 3.675, 3.905, 4.165]),
        'sdlog': np.array([0.12, 0.12, 0.11, 0.17])
    }
}


# --- 函数定义 ---

def ds_given_w(w, meanlog_params, sdlog_params):
    scales = np.exp(meanlog_params)
    # scale = np.exp(scales)
    F1 = lognorm.cdf(w, s=sdlog_params[0], scale=np.exp(scales[0]))
    F2 = lognorm.cdf(w, s=sdlog_params[1], scale=np.exp(scales[1]))
    F3 = lognorm.cdf(w, s=sdlog_params[2], scale=np.exp(scales[2]))
    F4 = lognorm.cdf(w, s=sdlog_params[3], scale=np.exp(scales[3]))

    probs = np.zeros(5)
    probs[0] = 1 - F1
    last_valid_F = F1

    # Re-anchoring logic
    for i, current_F in enumerate([F2, F3, F4], start=1):
        p = last_valid_F - current_F
        if p < 0:
            probs[i] = 0
        else:
            probs[i] = p
            last_valid_F = current_F

    probs[4] = last_valid_F
    return probs


def fw_unif(w, lower, upper):
    if lower <= w <= upper:
        return 1 / (upper - lower)
    return 0


def limiting_pmf(meanlog_params, sdlog_params, lower, upper, rel_tol=1e-8):
    pk = np.zeros(5)
    for k in range(5):
        def integrand(w):
            return ds_given_w(w, meanlog_params, sdlog_params)[k] * fw_unif(w, lower, upper)

        pk[k], _ = quad(integrand, lower, upper, epsrel=rel_tol)
    return {"DS0": pk[0], "DS1": pk[1], "DS2": pk[2], "DS3": pk[3], "DS4": pk[4]}


def plot_cdf(ef_label, meanlog_params, sdlog_params, lower, upper, arch_name, folder_name, n=500):
    xmax = upper
    x = np.linspace(0, xmax, n)

    scales = np.exp(meanlog_params)
    F1 = lognorm.cdf(x, s=sdlog_params[0], scale=np.exp(scales[0]))
    F2 = lognorm.cdf(x, s=sdlog_params[1], scale=np.exp(scales[1]))
    F3 = lognorm.cdf(x, s=sdlog_params[2], scale=np.exp(scales[2]))
    F4 = lognorm.cdf(x, s=sdlog_params[3], scale=np.exp(scales[3]))

    plt.figure(figsize=(8, 6))
    plt.plot(x, F1, lw=2, color="blue", label="LS1")
    plt.plot(x, F2, lw=2, color="red", label="LS2")
    plt.plot(x, F3, lw=2, color="green", label="LS3")
    plt.plot(x, F4, lw=2, color="purple", label="LS4")

    plt.axvline(x=lower, linestyle="--", color="black")
    plt.axvline(x=upper, linestyle="--", color="black")

    plt.xlabel("Windspeed (m/s)", fontsize=14)
    plt.ylabel("P(DS >= k | w)", fontsize=14)
    plt.title(f"{arch_name.upper()} - {ef_label}", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-0.02, 1.08)

    file_path = os.path.join(folder_name, f"pmf_plot_{arch_name}_{ef_label}.png")
    plt.savefig(file_path)
    plt.close()


# --- 主程序运行区 ---

MPH_TO_MS = 0.44704
ef_ranges = {
    'EF0': (65.0 * MPH_TO_MS, 86.0 * MPH_TO_MS),
    'EF1': (86.0 * MPH_TO_MS, 111.0 * MPH_TO_MS),
    'EF2': (111.0 * MPH_TO_MS, 136.0 * MPH_TO_MS),
    'EF3': (136.0 * MPH_TO_MS, 166.0 * MPH_TO_MS),
    'EF4': (166.0 * MPH_TO_MS, 200.0 * MPH_TO_MS)
}

# 文件夹设置
output_folder = "output_pmf_plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

final_results_all_archetypes = {}

for arch_name, params in archetype_params.items():
    print(f"Processing Archetype: {arch_name.upper()}")
    meanlog = params['meanlog']
    sdlog = params['sdlog']
    arch_results = {}

    for ef_label, (lower, upper) in ef_ranges.items():
        plot_cdf(ef_label, meanlog, sdlog, lower, upper, arch_name, output_folder)
        pmf = limiting_pmf(meanlog, sdlog, lower, upper)
        arch_results[ef_label] = {key: round(value, 4) for key, value in pmf.items()}

    final_results_all_archetypes[arch_name] = arch_results

# --- 导出 CSV 文件 ---
csv_filename = "incore_pmf.csv"
header = ["Archetype", "EF_Level", "DS0", "DS1", "DS2", "DS3", "DS4"]

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    for arch_name, ef_data in final_results_all_archetypes.items():
        for ef_label, ds_probs in ef_data.items():
            row = [
                arch_name,
                ef_label,
                ds_probs["DS0"],
                ds_probs["DS1"],
                ds_probs["DS2"],
                ds_probs["DS3"],
                ds_probs["DS4"]
            ]
            writer.writerow(row)

print(f"\nResults have been exported to '{csv_filename}'.")