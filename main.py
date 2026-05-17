import os
import subprocess
import logging
import sys
import config


def main():
    """Orchestrates the execution of the entire data analysis pipeline."""
    if os.path.exists(config.PIPELINE_LOG_PATH):
        os.remove(config.PIPELINE_LOG_PATH)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.PIPELINE_LOG_PATH),
            logging.StreamHandler()
        ]
    )
    logging.info("--- Starting the Data Analysis Pipeline ---")

    scripts_to_run = [
        '1_dataCleaning.py',
        '2_dataPreprocessing_test.py',
        '3_mutual_info.py',
        '4_modeling.py',
        # 'run_best_seed_permutation.py', # comment for now to run faster
        '5_calculate_mcr.py',
        '6_generate_report.py',
    ]

    # Clone environment and set OpenBLAS fix for memory unallocation crash
    custom_env = os.environ.copy()
    custom_env["OPENBLAS_NUM_THREADS"] = "1"
    custom_env["OMP_NUM_THREADS"] = "1"
    custom_env["MKL_NUM_THREADS"] = "1"
    custom_env["VECLIB_MAXIMUM_THREADS"] = "1"
    custom_env["NUMEXPR_NUM_THREADS"] = "1"

    # 【修改部分】获取当前这个主脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"--- Working Directory set to: {current_dir} ---")

    for script in scripts_to_run:
        logging.info(f"--- Running: {script} ---")

        # 拼接子脚本的绝对路径，防止找不到脚本
        script_path = os.path.join(current_dir, script)

        result = subprocess.run(
            [sys.executable, script_path],  # 使用完整路径运行子脚本
            capture_output=True,
            text=True,
            env=custom_env,
            cwd=current_dir  # 【修改部分】使用相对推导出的动态目录
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            logging.error(f"--- Traceback in {script}:\n{result.stderr} ---")
            logging.error(f"--- Error in {script}. Pipeline stopped. ---")
            exit(1)
        logging.info(f"--- Finished: {script} ---")

    logging.info("--- Data Analysis Pipeline Finished Successfully ---")
    print(f"\n✅ Pipeline Complete! Update your Visual Summary to view the latest plots.\n")


if __name__ == '__main__':
    main()