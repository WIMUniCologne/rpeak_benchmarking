
# R-Peak Benchmarking

This repository is linked to the paper: **_Benchmarking QRS Detection Algorithms Across Diverse ECG Datasets Under Varying Noise Conditions_**.

It benchmarks multiple R-peak detection algorithms on several ECG databases. The benchmarking code expects **CSV evaluation files** under `data/csv/...`.

## Databases

Supported databases (identifiers used in `main.py` / `benchmarkplatform.py`):

- `MITAR`: MIT Arrhythmia Database
- `MITNST`: MIT Noise Stress Test Database
- `MITLT`: MIT Long Term Database
- `Fantasia`: Fantasia Database
- `PTT`: Pulse Transit Time Database

## Algorithms

Supported algorithms (identifiers used in `main.py` / `benchmarkplatform.py`):

- Classical/feature-based: `pan`, `hamilton`, `elgendi`, `shaik`, `park`, `arteagaFalconi`, `xu`, `nguyen`, `zhai`, `kumari`, `xia`
- Learned/DL: `zahid`, `laitala`, `han_cnn`, `han_rnn`, `xiang`, `celik`

## Data preparation

1. Place the original/raw database files under `data/raw/`.
2. Generate the CSV evaluation files:

```bash
# generate all supported databases
python -m filecreator --db all

# or generate a single database
python -m filecreator --db MITNST
python -m filecreator --db MITAR
python -m filecreator --db Fantasia
python -m filecreator --db MITLT
python -m filecreator --db PTT
```

This writes CSVs to `data/csv/<DatabaseName>/` and a `summary.txt` per database.

## Run benchmarking

Edit `databases_to_run` and `algos_to_run` in `main.py`, then run:

```bash
python main.py
```

Results are written to `results/<DB>_results.txt` and `results/overall_results.txt`.

## Notes

- `filecreator.py` is import-safe and can be used as a library (e.g., call `create_database_csvs("MITNST")`).
- If you also use the CPSC dataset helper, run `cpsc_to_csv.py` directly.

