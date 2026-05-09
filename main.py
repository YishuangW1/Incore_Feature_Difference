import os
import subprocess
import logging
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
        '2_dataPreprocessing.py',
        '3_mutual_info.py',
        '4_modeling.py',
        # 'run_best_seed_permutation.py', # comment for now to run faster
        '5_calculate_mcr.py', # New script for MCR calculation comment for now to run faster
        '8_generate_report.py', # comment for now to run faster
    ]

    import sys
    
    # Clone environment and set OpenBLAS fix for memory unallocation crash
    custom_env = os.environ.copy()
    custom_env["OPENBLAS_NUM_THREADS"] = "1"
    custom_env["OMP_NUM_THREADS"] = "1"
    custom_env["MKL_NUM_THREADS"] = "1"
    custom_env["VECLIB_MAXIMUM_THREADS"] = "1"
    custom_env["NUMEXPR_NUM_THREADS"] = "1"
    
    for script in scripts_to_run:
        logging.info(f"--- Running: {script} ---")
        result = subprocess.run(
            [sys.executable, script], 
            capture_output=True, 
            text=True, 
            env=custom_env,
            cwd='/Users/yishuang/PycharmProjects/incore_git_mcr/MayfieldSimulations'
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