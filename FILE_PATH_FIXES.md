# File Path Issues Fixed

## Problem Summary
The HPC submission script was failing with "Missing required files" errors even though you had provided direct paths in the Python script. This was because:

1. **HPC script validation** was looking for files in old/incorrect locations
2. **Python script configuration** needed to be updated to look in the right directories where your actual files exist
3. **Path mismatches** between what the HPC script expected vs. where files actually were

## Root Cause Analysis

### Original Issue
```bash
ERROR: Missing required files:
  ✗ feature selection file (one of: selected_features._finalcsv feature_list_final.csv)
  ✗ label file (one of: y_label.csv label_union.csv Selected_label_sources.csv)
```

### Actual File Locations Found
```bash
✓ deterministic_build/selected_features_final.csv   # From build_labels_and_features.py
✓ engineered/feature_list_final.csv                # From engineer_features.py  
✓ engineered/y_label.csv                           # From engineer_features.py
✓ deterministic_build/label_union.csv              # From build_labels_and_features.py
✓ deterministic_build/selected_label_sources.csv   # From build_labels_and_features.py
```

## Fixes Applied

### 1. Updated Python Script File Discovery (`determinstic_fe_build.py`)

**Before:**
```python
SELECTED_FEATURES_FILES = [
    "selected_features_finalcsv",          # Wrong filename
    "feature_list_final.csv"               # Wrong location
]
LABEL_FILES = ["y_label.csv", "label_union.csv"]  # Wrong locations
```

**After:**
```python
SELECTED_FEATURES_FILES = [
    "deterministic_build/selected_features_final.csv",  # from build script
    "engineered/feature_list_final.csv",               # from engineer script  
    "selected_features_finalcsv",                       # fallback
    "feature_list_final.csv"                           # fallback
]
LABEL_FILES = [
    "engineered/y_label.csv",                          # from engineer script
    "deterministic_build/label_union.csv",             # from build script
    "y_label.csv", "label_union.csv"                   # fallbacks
]
```

### 2. Updated HPC Script Validation (`submit_deterministic_job.sh`)

**Before:**
```bash
feature_files=(
    "selected_features._finalcsv"    # Wrong filename
    "feature_list_final.csv"         # Wrong location
)
label_files=(
    "y_label.csv"                    # Wrong location
    "label_union.csv"                # Wrong location  
    "Selected_label_sources.csv"     # Wrong location
)
```

**After:**
```bash
feature_files=(
    "selected_features_finalcsv"
    "deterministic_build/selected_features_final.csv"
    "engineered/feature_list_final.csv"
    "feature_list_final.csv"
)
label_files=(
    "engineered/y_label.csv"
    "deterministic_build/label_union.csv"
    "deterministic_build/selected_label_sources.csv"
    "y_label.csv"
    "label_union.csv"
    "Selected_label_sources.csv"
)
```

### 3. Fixed Output File Paths

Updated all output file references in the HPC script to use absolute paths:
```bash
/home/vhsingh/Parshvi_project/engineered/X_features.parquet
/home/vhsingh/Parshvi_project/engineered/y_label.csv
# etc.
```

### 4. Added Directory Creation
```python
# Ensure output directory exists
Path("/home/vhsingh/Parshvi_project/engineered").mkdir(exist_ok=True, parents=True)
```

## Verification

### File Discovery Test Results
```bash
✅ SUCCESS: All required files found! Script should run without file errors.

Testing file discovery...
Selected features file: deterministic_build/selected_features_final.csv
Label files available: ['engineered/y_label.csv', 'deterministic_build/label_union.csv']
Guard files available: ['deterministic_build/guard_set.txt', 'do_not_use_features.txt']
Build summary files: ['deterministic_build/build_summary.json', 'engineered/best_config_used.json']
```

### Script Execution Test
- ✅ Python script runs successfully
- ✅ Output files created in `/home/vhsingh/Parshvi_project/engineered/`
- ✅ HPC script validation passes

## File Workflow Understanding

Your project has this workflow structure:

1. **`build_labels_and_features.py`** → Creates files in `deterministic_build/`
   - `selected_features_final.csv`
   - `label_union.csv` 
   - `selected_label_sources.csv`
   - `guard_set.txt`
   - `build_summary.json`

2. **`engineer_features.py`** → Creates files in `engineered/`
   - `feature_list_final.csv`
   - `y_label.csv`
   - `best_config_used.json`

3. **`determinstic_fe_build.py`** → Reads from both directories, outputs to `engineered/`
   - Reads from: `deterministic_build/` and `engineered/`
   - Writes to: `engineered/` (absolute paths)

## Resolution
The "missing files" error was not because files didn't exist, but because the validation logic was looking in the wrong places. Both the HPC script validation and Python script file discovery have been updated to:

1. **Search in the correct directories** where your pipeline actually creates files
2. **Use proper fallback order** (primary locations first, then fallbacks)
3. **Handle both relative and absolute paths** appropriately
4. **Match your actual workflow** and file organization

The script should now run successfully on HPC without file path issues.
