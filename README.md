# Solid-Polymer-Electrolytes-in-Lithium-Metal-Batteries

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains machine learning workflows and tools for pridicting the propertice of solvents for SSEs. Below are key details for using this repository.

---

## Table of Contents
- [Algorithms and Code Usage](#algorithms-and-code-usage)
- [Interpretability Analysis](#interpretability-analysis)
- [Dataset Handling](#dataset-handling)
- [Requesting Model Training Data](#requesting-model-training-data)
- [Contributing](#contributing)
- [License](#license)

---

## Algorithms and Code Usage

### Implemented Algorithms
This project utilizes the following algorithms:
- **CGCNN** (Crystal Graph Convolutional Neural Network)  
  For usage, refer to the official repository: [txie-93/cgcnn](https://github.com/txie-93/cgcnn).
- **XGBoost**  
  Install via `pip install xgboost` and refer to the [official documentation](https://xgboost.readthedocs.io/).
- **SISSO** (Sure Independence Screening and Sparsifying Operator)  
  Input files (`SISSO.in`, etc.) are provided in the `SISSO/` directory. For guidance, see [rouyang2017/SISSO](https://github.com/rouyang2017/SISSO).

---

## Interpretability Analysis
To reproduce the interpretability analysis:
1. Install Jupyter for data visualization:  
   ```bash
   pip install jupyterlab

2. Navigate to the interpretability/ directory and run
   ```bash
   jupyter notebook
   
3. Open the provided .ipynb files for interactive visualization.

---

## Dataset Handling
### Data Acquisition

Scripts to acquire and process datasets are provided in datasets/:

1. Fetch data from Chemspider:
Run 1.Data_acquired_from_chemspider.py (requires API access).

3. Convert JSON to MOL files:
Use 2.Convert_json_to_mol.py.

### High-Throughput DFT Calculations
Scripts 3.1_gaussian_htdft.py and 3.2_gaussian_htdft.py contain parameters for DFT calculations using Gaussian (commercial software). 
Adjust computational parameters as needed.

---

## Requesting Model Training Data
The datasets used for model training are not publicly redistributable due to licensing constraints. To request access:

1. Contact the author at mejiadongs@ust.hk with a brief description of your research purpose.
2. Data will be shared under a mutually agreed license.

---

## Contributing
Contributions are welcome! Open an issue or submit a pull request for improvements.

---
## License
This project is licensed under the MIT License. See LICENSE for details.
