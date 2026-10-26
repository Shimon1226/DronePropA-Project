from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
import re

# =========================================================
# DRONEPROPA: DRONE HEALTH MONITORING AND FAULT DETECTION
# =========================================================

# Dataset folder
DATASET_FOLDER = Path(
    r"C:\Users\Shimon Pandey\Downloads\SML PROJECT\DronePropA Motion Trajectories Dataset for Commercial Drones with Defective Propellers\DronePropA Motion Trajectories Dataset"
)

# Save the output in the project folder
OUTPUT_FILE = Path("drone_features.csv")

# Documented QDrone_data row indices.
# These are 1-based indices from the dataset documentation.
# Python uses 0-based indexing, so we subtract 1 below.
SELECTED_ROWS = {
    2: "measured_roll",
    3: "measured_pitch",
    27: "measured_roll_rate_1",
    28: "measured_pitch_rate_1",
    29: "measured_yaw_rate_1",
    30: "acceleration_x_1",
    31: "acceleration_y_1",
    32: "acceleration_z_1",
    33: "measured_roll_rate_2",
    34: "measured_pitch_rate_2",
    35: "measured_yaw_rate_2",
    36: "acceleration_x_2",
    37: "acceleration_y_2",
    38: "acceleration_z_2",
    46: "height",
    47: "front_left_motor",
    48: "front_left_esc",
    49: "front_right_motor",
    50: "front_right_esc",
    51: "back_left_motor",
    52: "back_left_esc",
    53: "back_right_motor",
    54: "back_right_esc",
}


def extract_signal_features(signal):
    """Calculate statistical features from one telemetry signal."""

    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "rms": np.nan,
        }

    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
    }


def process_flight(file_path):
    """Load one MATLAB flight and extract its features."""

    # Identify the fault group from the filename
    match = re.match(r"F([0-3])_", file_path.name)

    if match is None:
        print(f"Skipping unrecognized filename: {file_path.name}")
        return None

    fault_group = int(match.group(1))

    # Label: 0 = Healthy, 1 = Faulty
    label = 0 if fault_group == 0 else 1

    # Load the MATLAB file
    data = loadmat(file_path)

    if "QDrone_data" not in data:
        print(f"Missing QDrone_data: {file_path.name}")
        return None

    qdrone_data = np.asarray(data["QDrone_data"])

    # Check that the expected telemetry rows exist
    if qdrone_data.ndim != 2:
        print(f"Unexpected data shape: {file_path.name}")
        return None

    features = {}

    # Extract statistical features from documented signals
    for row_number, signal_name in SELECTED_ROWS.items():

        # Convert documented 1-based row number to Python index
        row_index = row_number - 1

        if row_index >= qdrone_data.shape[0]:
            continue

        signal = qdrone_data[row_index, :]

        stats = extract_signal_features(signal)

        for stat_name, value in stats.items():
            feature_name = f"{signal_name}_{stat_name}"
            features[feature_name] = value

    # Add flight information
    features["filename"] = file_path.name
    features["fault_group"] = fault_group
    features["label"] = label

    return features


def main():

    print("=" * 60)
    print("DRONE HEALTH MONITORING AND PROPELLER FAULT DETECTION")
    print("=" * 60)

    # Check dataset folder
    if not DATASET_FOLDER.exists():
        print("\nERROR: Dataset folder not found.")
        print("Check the DATASET_FOLDER path in main.py.")
        return

    # Find all MATLAB files
    mat_files = sorted(DATASET_FOLDER.glob("*.mat"))

    print(f"\nDataset folder: {DATASET_FOLDER}")
    print(f"MATLAB files found: {len(mat_files)}")

    if not mat_files:
        print("ERROR: No .mat files found.")
        return

    # Process every flight
    all_flights = []
    failed_files = []

    for number, file_path in enumerate(mat_files, start=1):

        try:
            result = process_flight(file_path)

            if result is not None:
                all_flights.append(result)
            else:
                failed_files.append(file_path.name)

        except Exception as error:
            print(f"\nError processing {file_path.name}: {error}")
            failed_files.append(file_path.name)

        if number % 10 == 0 or number == len(mat_files):
            print(f"Processed {number}/{len(mat_files)} files")

    # Check results
    if not all_flights:
        print("\nERROR: No flight features were extracted.")
        return

    # Create one row per flight
    df = pd.DataFrame(all_flights)

    # Put identification columns at the beginning
    identification_columns = ["filename", "fault_group", "label"]
    feature_columns = [
        column
        for column in df.columns
        if column not in identification_columns
    ]

    df = df[identification_columns + feature_columns]

    # Replace infinite values with missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Save the complete dataset
    df.to_csv(OUTPUT_FILE, index=False)

    # Display summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print(f"\nOutput file: {OUTPUT_FILE.resolve()}")
    print(f"Successfully processed: {len(df)}")
    print(f"Failed or skipped files: {len(failed_files)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index().rename(
        index={0: "Healthy (0)", 1: "Faulty (1)"}
    ))

    print("\nFault group distribution:")
    print(df["fault_group"].value_counts().sort_index())

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nSUCCESS: drone_features.csv has been created!")

    if failed_files:
        print("\nFiles requiring investigation:")
        for filename in failed_files:
            print(filename)


if __name__ == "__main__":
    main()