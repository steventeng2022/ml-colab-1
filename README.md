# NFL Big Data Bowl 2027 — Beyond the stopwatch

A reproducible receiver study asking whether sensor-measured speed during the Combine 40-yard dash adds information beyond official measurements when evaluating rookie NFL route movement.

[View the notebook](nfl_big_data_bowl_2027.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/steventeng2022/ml-colab-1/blob/main/nfl_big_data_bowl_2027.ipynb) · [Competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)

## Run in Kaggle

1. Download this notebook or import it into a Kaggle Notebook.
2. Attach **NFL Big Data Bowl 2027** using **Add Input**.
3. Select CPU and **Run All**. Internet access is unnecessary once the data is attached.

Results are written to `/kaggle/working/receiver_speed`.

## Run in Colab or locally

Obtain access to the official competition data on Kaggle and extract its CSV files into one folder. In Colab, upload that folder or mount the Drive folder containing it.

For a local environment, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the notebook and set `DATA_DIR_OVERRIDE` in its configuration cell to the extracted competition folder. Run all cells. Local results are written to `receiver_speed_outputs/`.

The game tracking files are large. CPU is sufficient; allow time for the first scan. A source-validated cache accelerates reruns in the same session.

## Analysis

- Header and drill-label detection, duplicate-frame checks, and explicit coverage diagnostics.
- Three-frame speed smoothing and equal rookie-season exposure windows.
- Four baseline/model comparisons trained on 2023–2024, with 2025 held out.
- A route-context sensitivity using route type and release duration, fitted on training players.
- Paired bootstrap intervals, four PNG/SVG figures, and a computed research summary.
- Explicit skipped-model reports when fixed sample-size requirements are not met.

The notebook keeps primary, sensitivity, and exploratory outcomes distinct. The analysis is observational; route movement is not an overall talent score.

## Validation and results

Notebook structure and Python syntax were checked. All code cells were exercised on generated fixtures, covering drill-column variants, duplicate timestamps across chunks, cache consistency, missing events, and empty cohorts.

The official competition dataset has not been executed locally. Run the notebook on the official data before interpreting or reporting findings. Stored notebook outputs are empty.

## Data and attribution

Competition data is provided by The National Football League through Kaggle. Obtain it from the [official data page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data) and follow its license and competition rules.

This repository contains the analysis code. A competition entry also needs a reviewed write-up and an executed notebook, following the [official submission requirements](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/overview).
