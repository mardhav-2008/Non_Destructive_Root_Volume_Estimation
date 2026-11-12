import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import load_observation_image
from models.cnn_regressor import RootVolumeRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CSV = ROOT / "Test.csv"
IMAGE_ROOT = ROOT / "data" / "test"

CHECKPOINT_PATH = ROOT / "outputs" / "uncertainty" / "image_fixed_epoch.pt"

METRICS_PATH = ROOT / "outputs" / "uncertainty" / "image_metrics.json"


# ============================================================
# LOAD MODEL AND CALIBRATION
# ============================================================


def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}\n\n"
            "Run scripts/calibrate_predictions.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = RootVolumeRegressor(
        category_sizes=checkpoint["category_sizes"],
        mode="image",
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


def load_calibration_threshold():

    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH) as file:
        metrics = json.load(file)

    thresholds = metrics.get("calibration_thresholds", {})

    threshold = thresholds.get("observation_q")

    if threshold is None or not np.isfinite(threshold):
        return None

    return float(threshold)


# ============================================================
# APPLICATION
# ============================================================


class RootVolumeApp:
    def __init__(self, root, model, calibration_q):

        self.root = root
        self.model = model
        self.calibration_q = calibration_q

        self.root.title("Root Volume Estimation | G14")

        self.root.geometry("950x720")
        self.root.minsize(800, 600)

        self.frame = pd.read_csv(TEST_CSV)

        self.selected_image = None

        self.setup_style()
        self.setup_ui()

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    def setup_style(self):

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "Title.TLabel",
            font=("Arial", 20, "bold"),
        )

        style.configure(
            "Subtitle.TLabel",
            font=("Arial", 10),
        )

        style.configure(
            "Result.TLabel",
            font=("Arial", 16, "bold"),
        )

        style.configure(
            "TButton",
            font=("Arial", 10),
            padding=8,
        )

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def setup_ui(self):

        main = ttk.Frame(
            self.root,
            padding=20,
        )

        main.pack(
            fill="both",
            expand=True,
        )

        # Header
        ttk.Label(
            main,
            text="Root Volume Estimator",
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            main,
            text="Non-Destructive Root Volume Estimation | G14",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 15))

        ttk.Separator(
            main,
            orient="horizontal",
        ).pack(fill="x", pady=5)

        # Observation selection
        selection_frame = ttk.LabelFrame(
            main,
            text="Observation Selection",
            padding=12,
        )

        selection_frame.pack(
            fill="x",
            pady=10,
        )

        self.observation_var = tk.StringVar()

        self.observation_dropdown = ttk.Combobox(
            selection_frame,
            textvariable=self.observation_var,
            values=self.frame["ID"].astype(str).tolist(),
            state="readonly",
            width=35,
        )

        self.observation_dropdown.pack(
            side="left",
            padx=(0, 10),
        )

        self.observation_dropdown.bind(
            "<<ComboboxSelected>>",
            self.on_selection,
        )

        ttk.Button(
            selection_frame,
            text="Load Observation",
            command=self.on_selection,
        ).pack(side="left")

        # Metadata
        self.metadata_label = ttk.Label(
            main,
            text="Select an observation to begin.",
            justify="left",
        )

        self.metadata_label.pack(
            anchor="w",
            pady=10,
        )

        # Scan preview
        image_frame = ttk.LabelFrame(
            main,
            text="Stacked Root Scan",
            padding=10,
        )

        image_frame.pack(
            fill="both",
            expand=True,
            pady=10,
        )

        self.image_label = ttk.Label(
            image_frame,
            text="No scan loaded",
            anchor="center",
        )

        self.image_label.pack(
            fill="both",
            expand=True,
        )

        # Prediction
        result_frame = ttk.LabelFrame(
            main,
            text="Prediction",
            padding=15,
        )

        result_frame.pack(
            fill="x",
            pady=10,
        )

        self.prediction_label = ttk.Label(
            result_frame,
            text="Root volume: --",
            style="Result.TLabel",
        )

        self.prediction_label.pack(
            anchor="w",
        )

        self.interval_label = ttk.Label(
            result_frame,
            text="Prediction interval: --",
        )

        self.interval_label.pack(
            anchor="w",
            pady=(8, 0),
        )

        self.status_label = ttk.Label(
            result_frame,
            text="Select an observation and run prediction.",
        )

        self.status_label.pack(
            anchor="w",
            pady=(8, 0),
        )

        # Action button
        ttk.Button(
            main,
            text="Estimate Root Volume",
            command=self.predict,
        ).pack(
            anchor="e",
            pady=5,
        )

        # Footer
        ttk.Label(
            main,
            text=(
                "Experimental model — estimates and intervals "
                "are not validated for scientific or breeding decisions."
            ),
            style="Subtitle.TLabel",
        ).pack(
            anchor="w",
            pady=(10, 0),
        )

        if self.frame.empty:
            self.status_label.config(text="Test.csv contains no observations.")
        else:
            self.observation_dropdown.current(0)
            self.on_selection()

    # --------------------------------------------------------
    # SELECT OBSERVATION
    # --------------------------------------------------------

    def get_selected_row(self):

        observation_id = self.observation_var.get()

        if not observation_id:
            return None

        matches = self.frame[self.frame["ID"].astype(str) == observation_id]

        if matches.empty:
            return None

        return matches.iloc[0]

    def on_selection(self, event=None):

        row = self.get_selected_row()

        if row is None:
            return

        metadata = (
            f"ID: {row['ID']}\n"
            f"Folder: {row['FolderName']}\n"
            f"Plant: {row['PlantNumber']}\n"
            f"Genotype: {row['Genotype']}\n"
            f"Stage: {row['Stage']}\n"
            f"Side: {row['Side']}\n"
            f"Scan range: {row['Start']} – {row['End']}"
        )

        self.metadata_label.config(text=metadata)

        try:
            image = load_observation_image(
                image_root=IMAGE_ROOT,
                folder=str(row["FolderName"]),
                side=str(row["Side"]),
                start=int(row["Start"]),
                end=int(row["End"]),
            )

            # Channel 0 contains grayscale intensity.
            grayscale = image[0].cpu().numpy()

            grayscale = (np.clip(grayscale, 0, 1) * 255).astype(np.uint8)

            pil_image = Image.fromarray(
                grayscale,
                mode="L",
            )

            pil_image.thumbnail(
                (700, 300),
                Image.Resampling.LANCZOS,
            )

            self.selected_image = ImageTk.PhotoImage(pil_image)

            self.image_label.config(
                image=self.selected_image,
                text="",
            )

            self.prediction_label.config(text="Root volume: --")

            self.interval_label.config(text="Prediction interval: --")

            self.status_label.config(text="Scan loaded. Ready for prediction.")

        except Exception as error:
            self.image_label.config(
                image="",
                text="Unable to load scan.",
            )

            self.selected_image = None

            messagebox.showerror(
                "Image loading error",
                str(error),
            )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self):

        row = self.get_selected_row()

        if row is None:
            messagebox.showwarning(
                "No observation",
                "Select an observation first.",
            )
            return

        try:
            image = load_observation_image(
                image_root=IMAGE_ROOT,
                folder=str(row["FolderName"]),
                side=str(row["Side"]),
                start=int(row["Start"]),
                end=int(row["End"]),
            )

            image = image.unsqueeze(0).to(DEVICE)

            self.model.eval()

            prediction = self.model(image=image)

            prediction = prediction.clamp_min(0)

            volume = float(prediction.item())

            self.prediction_label.config(text=f"Estimated root volume: {volume:.3f}")

            if self.calibration_q is not None:
                lower = max(
                    0.0,
                    volume - self.calibration_q,
                )

                upper = volume + self.calibration_q

                self.interval_label.config(
                    text=(f"90% target prediction interval: [{lower:.3f}, {upper:.3f}]")
                )

                self.status_label.config(
                    text=("Experimental observation-level conformal interval.")
                )

            else:
                self.interval_label.config(text="Prediction interval unavailable.")

                self.status_label.config(
                    text=(
                        "Calibration threshold not found. "
                        "Only the point estimate is available."
                    )
                )

        except Exception as error:
            messagebox.showerror(
                "Prediction error",
                str(error),
            )


# ============================================================
# ENTRY POINT
# ============================================================


def main():

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    try:
        model = load_model()

        calibration_q = load_calibration_threshold()

        app_root = tk.Tk()

        RootVolumeApp(
            app_root,
            model,
            calibration_q,
        )

        app_root.mainloop()

    except Exception as error:
        messagebox.showerror(
            "Application startup error",
            str(error),
        )


if __name__ == "__main__":
    main()
