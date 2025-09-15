# Master-Thesis-Code

## Requirements

- Python 3.10+ (tested with 3.11)
- Packages:
  ```bash
  pip install numpy pandas scikit-learn SALib matplotlib gurobipy

## Quickstart

### In repo root:
python main.py


## Project structure

main.py                 # entry point (performance + S1 sensitivity)
solver.py               # integer linear model (Gurobi)
generate_instances.py   # scenario generators (V1/V2/V3)
GBT.py                  # demand forecasting (data prep + Gradient Boosting Trees)
sensitivity_analysis.py # Morris sampling, evaluation, analysis, plotting
the_burger_spot.csv     # input data (row-level orders)
