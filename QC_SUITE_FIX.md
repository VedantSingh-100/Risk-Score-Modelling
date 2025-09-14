# QC Suite TypeError Fix

## Problem
```python
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

## Root Cause
The issue was a **type mismatch** in the QC suite script:

### Before (Problematic Code)
```python
OUTPUT_BASE_DIR = "/home/vhsingh/qc_outputs"  # String type
# ...
OUT_QC_NULLS = OUTPUT_BASE_DIR / "qc_nulls_report.csv"  # ❌ Error: str / str
```

The `/` operator for path concatenation **only works with `pathlib.Path` objects**, not strings.

## Solution
Convert the string to a `Path` object:

### After (Fixed Code)
```python
OUTPUT_BASE_DIR = Path("/home/vhsingh/qc_outputs")  # Path object
# ...
OUT_QC_NULLS = OUTPUT_BASE_DIR / "qc_nulls_report.csv"  # ✅ Works: Path / str
```

## Key Changes Made

### Line 12: Convert to Path Object
```python
# Before
OUTPUT_BASE_DIR = "/home/vhsingh/qc_outputs"

# After  
OUTPUT_BASE_DIR = Path("/home/vhsingh/qc_outputs")
```

### Line 13: Simplified Directory Check
```python
# Before
if not Path(OUTPUT_BASE_DIR).exists():
    Path(OUTPUT_BASE_DIR).mkdir(exist_ok=True, parents=True)

# After
if not OUTPUT_BASE_DIR.exists():
    OUTPUT_BASE_DIR.mkdir(exist_ok=True, parents=True)
```

## Verification

### Script Execution
```bash
✅ Script runs without errors
✅ All QC output files generated
```

### Output Files Created
```bash
/home/vhsingh/qc_outputs/
├── qc_nulls_report.csv           - Null value analysis
├── qc_guard_violations.csv       - Guard rule violations  
├── qc_redundancy_pairs.csv       - Feature redundancy analysis
├── qc_single_feature_metrics.csv - Individual feature metrics
└── qc_summary.md                 - Overall QC summary
```

## Python Path Operations Reference

### ✅ Correct Usage
```python
from pathlib import Path

# Path object + string
path = Path("/home/user") / "file.txt"

# Path object + Path object  
path = Path("/home/user") / Path("subdir/file.txt")

# Path object method
path = Path("/home/user").joinpath("file.txt")
```

### ❌ Incorrect Usage
```python
# String + string (doesn't work)
path = "/home/user" / "file.txt"  # TypeError

# Mixed types (inconsistent)
path = "/home/user" + "/" + "file.txt"  # Works but not recommended
```

## Best Practice
Always use `pathlib.Path` for path operations in modern Python:
- More readable and intuitive
- Cross-platform compatible
- Rich set of path manipulation methods
- Type-safe operations

The QC suite now runs successfully and generates comprehensive quality control reports for your feature engineering pipeline!
