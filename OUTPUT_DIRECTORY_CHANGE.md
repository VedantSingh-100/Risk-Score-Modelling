# Output Directory Change Summary

## Change Made
Moved all output files from the project directory to a separate dedicated directory.

## Old vs New Output Location

### Before
```
/home/vhsingh/Parshvi_project/engineered/
├── X_features.parquet
├── y_label.csv
├── feature_engineering_report_pre.csv
├── feature_engineering_report_post.csv
├── leakage_check.csv
├── transforms_config.json
└── dropped_features.json
```

### After
```
/home/vhsingh/deterministic_fe_outputs/
├── X_features.parquet
├── y_label.csv
├── feature_engineering_report_pre.csv
├── feature_engineering_report_post.csv
├── leakage_check.csv
├── transforms_config.json
└── dropped_features.json
```

## Benefits
1. **Clean separation**: Output files no longer clutter the project directory
2. **Easy management**: All outputs in one dedicated location
3. **Clear organization**: Separates code/config from generated outputs
4. **Better backup**: Can easily backup or exclude outputs from version control

## Files Modified

### 1. `determinstic_fe_build.py`
- Added `OUTPUT_BASE_DIR = "/home/vhsingh/deterministic_fe_outputs"`
- Updated all output file paths to use the new directory:
  - `REPORT_PRE`, `REPORT_POST`, `DROPPED_JSON`, `TRANSFORMS_JSON`
  - `LEAKAGE_CSV`, `X_OUT`, `Y_OUT`
- Added automatic directory creation
- Added debug output for better visibility

### 2. `submit_deterministic_job.sh`
- Updated output file validation to check new directory
- Updated feature matrix reporting paths
- Updated transformation summary paths
- All output references now point to `/home/vhsingh/deterministic_fe_outputs/`

## Verification Results

### Script Execution
```bash
=== Starting Deterministic Feature Engineering ===
Output directory: /home/vhsingh/deterministic_fe_outputs

Features kept: 53, rows: 49389
Artifacts written:
- /home/vhsingh/deterministic_fe_outputs/X_features.parquet
- /home/vhsingh/deterministic_fe_outputs/y_label.csv
- /home/vhsingh/deterministic_fe_outputs/feature_engineering_report_pre.csv
- /home/vhsingh/deterministic_fe_outputs/feature_engineering_report_post.csv
- /home/vhsingh/deterministic_fe_outputs/leakage_check.csv
- /home/vhsingh/deterministic_fe_outputs/transforms_config.json
- /home/vhsingh/deterministic_fe_outputs/dropped_features.json
```

### Output Files Created
```bash
✓ Output exists: X_features.parquet (11M)
✓ Output exists: y_label.csv (65K)  
✓ Output exists: transforms_config.json (33K)
✅ All expected outputs created in new directory
```

### HPC Validation
```bash
✓ Found feature file: deterministic_build/selected_features_final.csv
✓ Found label file: engineered/y_label.csv
✅ HPC validation passes - will find required input files
```

## Usage

### Local Execution
```bash
cd /home/vhsingh/Parshvi_project
python determinstic_fe_build.py
```
Outputs will be written to `/home/vhsingh/deterministic_fe_outputs/`

### HPC Execution  
```bash
cd /home/vhsingh/Parshvi_project
sbatch submit_deterministic_job.sh
```
The HPC script will automatically validate inputs and report on outputs in the new directory.

### Accessing Outputs
```bash
# View all outputs
ls -la /home/vhsingh/deterministic_fe_outputs/

# Load in Python
import pandas as pd
X = pd.read_parquet("/home/vhsingh/deterministic_fe_outputs/X_features.parquet")
y = pd.read_csv("/home/vhsingh/deterministic_fe_outputs/y_label.csv")["label"]
```

## Input Files (Unchanged)
The script still reads input files from their original locations:
- Raw data: `/home/vhsingh/Parshvi_project/50k_users_merged_data_userfile_updated_shopping.csv`
- Features: `/home/vhsingh/Parshvi_project/deterministic_build/selected_features_final.csv`
- Labels: `/home/vhsingh/Parshvi_project/engineered/y_label.csv`
- Guard: `/home/vhsingh/Parshvi_project/deterministic_build/guard_set.txt`
- Config: `/home/vhsingh/Parshvi_project/deterministic_build/build_summary.json`

Only the **output** location has changed to keep your project directory clean while maintaining access to all generated artifacts in a dedicated location.
