# Results Directory

This directory is intentionally kept lightweight.

Generated outputs are not committed by default. Run the experiment scripts to
create result files locally:

```powershell
python prog_fedmff.py --per-user-output results\fedmff_nyc_per_user.csv
python prog_mostpop.py --per-user-output results\mostpop_nyc_per_user.csv
python significance_test.py --ours results\fedmff_nyc_per_user.csv --baseline results\mostpop_nyc_per_user.csv
```

Do not place large generated CSV files, logs, model checkpoints, or downloaded
datasets in this directory when preparing a clean code package.
