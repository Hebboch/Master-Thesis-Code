# Master-Thesis-Code

## Requirements

- Python 3.10+ (tested with 3.11)
- Packages:
  ```bash
  pip install numpy pandas scikit-learn SALib matplotlib gurobipy
  
## Quick start

python main.py

## Project structure

| File | Description |
|---|---|
| `main.py`               | Entry point (performance + S1 sensitivity) |
| `solver.py`             | MILP model (Gurobi) |
| `generate_instances.py` | Scenario generators (V1/V2/V3) |
| `GBT.py`                | Demand forecasting (data prep + HGBR pipeline) |
| `sensitivity_analysis.py` | Morris sampling, evaluation, analysis, plotting |
| `the_burger_spot.csv`   | Input data (row-level orders) |


## Data

`the_burger_spot.csv` must be placed in the repository root.  
`GBT.py` reads it directly and derives the weekday from the `Date` column.
