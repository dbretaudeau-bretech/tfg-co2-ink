# Machine-learning readout of a colorimetric CO2-sensing ink

Code and results for the Treball de Fi de Grau of Daniel Bretaudeau Merce
(Facultat de Física, Universitat de Barcelona, June 2026).
Advisor: Ismael Benito Altamirano. Ink, chamber and measurement campaign by
Cristian Fàbrega and the EMERGE collaboration.

## What this is

A colorimetric CO2-sensing ink is read by four RGB+IR reflectance pixels
over a 77-hour gas-chamber protocol. The thesis characterises the ink
(sensitivity, humidity cross-sensitivity, response lag) and searches for a
readout that survives film drift: a ladder of models from least squares to
recurrent networks, the discovery that referencing the colour channels to
the co-measured infrared cancels most channel-common drift (a 12-feature
linear model reaches MAE 103 ppm / R^2 0.77 with no correction), and a
single per-illumination-change offset for the remaining factor of two
(50 ppm).

## Layout

- `code/` — full pipeline: `data4.py` (loaders, features, splits, audit),
  `run_night_ladder.py` + `run_ladder2-5.py` (model ladder, offset protocol,
  coarse-context LSTM, dead-reckoning), `run_idea1-4.py` (kinetic features,
  IR normalization, PatchTST, adversarial RH decoupling), `build_*.py`
  (figures), `verify_*.py` (numerical provenance checks).
- `results/` — metrics for every experiment (JSON), experiment notes, and
  `PROVENANCE_*.md`: every numerical claim of the memoria traced to a file
  and a recomputation.
- `figures/` — the memoria figures (PNG).
- `memoria/` — LaTeX source of the report (official UB Física template).

## Data

The raw chamber dataset belongs to the EMERGE collaboration and is not
redistributed here; it is available on reasonable request. The pipeline
expects `unified_5s_corrected.csv` (a 5-s-grid merge of the chamber log and
sensor CSVs) in the repository root; `data4.py` documents the expected
columns.

## Environment

Python 3.13, numpy/pandas/scikit-learn, PyTorch (CUDA) for the neural
models. All linear/tree models run in minutes on CPU; the LSTM variants
take ~10 min on one GPU.
