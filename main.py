from pathlib import Path
from scipy.io import loadmat

# Path to your downloaded dataset
DATASET_FOLDER = Path(
    r"C:\Users\Shimon Pandey\Downloads\SML PROJECT\DronePropA Motion Trajectories Dataset for Commercial Drones with Defective Propellers\DronePropA Motion Trajectories Dataset"
)

# Find MATLAB files
mat_files = sorted(DATASET_FOLDER.glob("*.mat"))

print("Dataset folder exists:", DATASET_FOLDER.exists())
print("Number of MAT files:", len(mat_files))

if mat_files:
    file_path = mat_files[0]
    data = loadmat(file_path)

    print("\nLoaded file:", file_path.name)
    print("\nVariables:")

    for key, value in data.items():
        if not key.startswith("__"):
            print(key, value.shape)
else:
    print("Dataset not found. Check the folder path.")