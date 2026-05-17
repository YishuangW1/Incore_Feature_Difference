import os
import logging
import sys
import pandas as pd
from fpdf import FPDF
from PIL import Image
import config


# --- Logging Configuration ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
                        force=True)


setup_logging()


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Analysis Report - Structural Damage Research', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_image_to_page(self, image_path, title):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            logging.warning(f"Could not open image {image_path}: {e}")
            return

        # Set landscape or portrait orientation based on image dimensions
        orientation = 'L' if width > height else 'P'
        self.add_page(orientation=orientation)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')

        # Automatically scale image to fit the page
        if orientation == 'L':
            self.image(image_path, x=10, y=25, w=275)
        else:
            self.image(image_path, x=10, y=25, w=190)


def find_file(filename, search_dirs):
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def main():
    logging.info("--- Generating Final PDF Report ---")

    # Define search directories
    search_directories = [
        config.BASE_RESULTS_DIR,
        config.SHAP_RESULTS_DIR,
        'top_feature_plots',
        'eda_results',
        '.'
    ]

    # Define image sequence in the report
    ordered_files = [
        # 1. Data Overview and Preprocessing Results
        'target_distribution_plots.png',
    ]

    # 2. Mutual Information Analysis (Dynamic Search)
    mi_plots = [f for f in os.listdir(config.BASE_RESULTS_DIR) if
                f.startswith('feature_importance_mi_') and f.endswith('.png')]
    # Sort to put 'original' first, then thresholds
    mi_plots.sort(key=lambda x: (0 if 'original' in x else 1, x))
    ordered_files.extend(mi_plots)

    # 3. Model Comparison and Performance
    ordered_files.append('model_comparison_main_metric.png')

    # 4. Automatically add prediction comparison plots for the top models
    try:
        perf_df = pd.read_csv(config.DETAILED_RESULTS_CSV)
        sort_metric = 'Test R2' if config.PROBLEM_TYPE == 'regression' else 'Test F1 (Weighted)'

        if sort_metric in perf_df.columns:
            top_models = perf_df.sort_values(by=sort_metric, ascending=False).head(3)
            for _, row in top_models.iterrows():
                combo_key = f"{row['Model']}_{row['Feature Set Name']}"
                fname = f"actual_vs_predicted_{combo_key.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                ordered_files.append(fname)
    except Exception as e:
        logging.warning(f"Could not add dynamic model plots: {e}")

    # 5. SHAP Explanation Results
    ordered_files.extend([
        'shap_summary_bar.png',
        'shap_beeswarm_plot.png'
    ])

    # 6. Collect feature relationship plots
    if os.path.exists('top_feature_plots'):
        rel_plots = [f for f in os.listdir('top_feature_plots') if f.endswith('.png')]
        ordered_files.extend(rel_plots)

    # Generate PDF
    pdf = PDF()
    found_any = False

    for filename in ordered_files:
        path = find_file(filename, search_directories)
        if path:
            # Convert filename to a more readable title
            title = filename.replace('.png', '').replace('_', ' ').title()
            pdf.add_image_to_page(path, title)
            found_any = True
            logging.info(f"Added {filename} to report.")
        else:
            logging.warning(f"File not found: {filename}")

    if found_any:
        # Fix: Save to standard reports directory instead of nested inside results
        os.makedirs(config.REPORT_DIR, exist_ok=True)
        output_path = os.path.join(config.REPORT_DIR, config.REPORT_FILENAME)
        pdf.output(output_path)
        logging.info(f"✅ Success! Report generated at: {output_path}")
    else:
        logging.error("No images found to generate a report.")


if __name__ == "__main__":
    main()