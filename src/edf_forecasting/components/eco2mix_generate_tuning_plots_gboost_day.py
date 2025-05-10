import os
import optuna
import logging

from optuna.visualization import plot_optimization_history, plot_param_importances

logging.basicConfig(level=logging.INFO)

def generate_tuning_plots(study_path: str, output_dir: str):
    """Charge une Optuna study et génère les visualisations (.png)."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        study = optuna.load_study(study_name="xgb_tuning", storage=f"sqlite:///{study_path}")

        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(output_dir, "optimization_history.png"))

        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(output_dir, "param_importances.png"))

        logging.info("Plots generated successfully.")
        return {
            "optimization_history": os.path.join(output_dir, "optimization_history.png"),
            "param_importances": os.path.join(output_dir, "param_importances.png")
        }

    except Exception as e:
        logging.warning(f"Failed to generate tuning plots: {e}")
        return {}
