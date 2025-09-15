#!/usr/bin/env bash
set -euo pipefail

echo "Starting repository relayout..."

# Create tree (already done, but ensure it exists)
mkdir -p configs data/raw data/interim data/processed src/preprocessing src/models src/utils scripts docs artifacts/models artifacts/reports artifacts/logs

# Move canonical configs
echo "Moving configuration files..."
for f in sweep/best_config_used.json best_config_used.json; do
  [[ -f "$f" ]] && mv -f "$f" configs/best_config_used.json && echo "✓ Moved $f to configs/" || true
done
for f in build_summary.json sweep/build_summary.json; do
  [[ -f "$f" ]] && mv -f "$f" configs/build_summary.json && echo "✓ Moved $f to configs/" || true
done
[[ -f label_policy.json ]] && mv -f label_policy.json configs/ && echo "✓ Moved label_policy.json to configs/" || true
[[ -f transforms_config.json ]] && mv -f transforms_config.json configs/ && echo "✓ Moved transforms_config.json to configs/" || true
for f in guard_Set.txt guard_set.txt do_not_use_features.txt leakage_guard_list.csv; do
  [[ -f "$f" ]] && mv -f "$f" configs/guard_Set.txt && echo "✓ Moved $f to configs/guard_Set.txt" && break || true
done
for f in Selected_label_sources.csv smart_label_candidates.csv negative_pattern_variables.csv; do
  [[ -f "$f" ]] && mv -f "$f" configs/Selected_label_sources.csv && echo "✓ Moved $f to configs/Selected_label_sources.csv" && break || true
done

# Move processed data / QC
echo "Moving processed data and QC files..."
for f in X_features.parquet y_label.csv; do
  [[ -f "data/$f" ]] && mv -f "data/$f" data/processed/ && echo "✓ Moved data/$f to data/processed/" || true
  [[ -f "$f" ]] && mv -f "$f" data/processed/ && echo "✓ Moved $f to data/processed/" || true
done
for f in qc_summary.md qc_single_feature_metrics.csv qc_single_feature_matrix.csv qc_redundancy_pairs.csv qc_redundancyt_pairs.csv leakage_check.csv feature_engineering_report_pre.csv feature_engineering_report_post.csv; do
  [[ -f "$f" ]] && mv -f "$f" data/processed/ && echo "✓ Moved $f to data/processed/" || true
done

# Move docs
echo "Moving documentation files..."
for f in MODEL_CARD.md model_card.md EXECUTIVE_SUMMARY.md final_enhanced_report.md final_enhanced_summary.py TRAINING_PIPELINE_ENHANCED.md TRAINING_RESULTS_EXPLAINED.md; do
  [[ -f "$f" ]] && mv -f "$f" docs/ && echo "✓ Moved $f to docs/" || true
done

# Move training code into src/models
echo "Moving training scripts..."
for f in train.py train_advanced.py; do
  [[ -f "$f" ]] && mv -f "$f" src/models/ && echo "✓ Moved $f to src/models/" || true
done

# Put big/one-off raw files to data/raw
echo "Moving raw data files..."
for f in 50k_users_merged_data_userfile_updated_shopping.csv Internal_Algo360VariableDictionary_WithExplanation.xlsx *.parquet *.pkl; do
  [[ -f "$f" ]] && mv -f "$f" data/raw/ && echo "✓ Moved $f to data/raw/" || true
done

# Stash experiments / sweeps / logs under artifacts
echo "Moving artifacts and logs..."
[[ -d sweep ]] && mv -f sweep artifacts/reports/sweep && echo "✓ Moved sweep/ to artifacts/reports/" || true
[[ -d logs  ]] && mv -f logs  artifacts/logs/old_logs && echo "✓ Moved logs/ to artifacts/logs/old_logs/" || true
[[ -d model_outputs ]] && mv -f model_outputs artifacts/reports/model_outputs && echo "✓ Moved model_outputs/ to artifacts/reports/" || true

# Create __init__.py files for Python packages
touch src/__init__.py src/preprocessing/__init__.py src/models/__init__.py src/utils/__init__.py

echo "✅ Repo relayout complete!"
echo ""
echo "Directory structure:"
find . -type d -name "__pycache__" -prune -o -type d -print | head -20 | sort

