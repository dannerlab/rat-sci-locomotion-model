# rat-sci-locomotion-model
Simulation code, model files, and data analysis of results presented in:

Shevtosva NA, Lockhart AJ, Rybak IA, Magnuson DSK, Danner SM. **Reorganization of spinal neural connectivity following recovery after thoracic spinal cord injury: insights from computational modelling.** *bioRxiv*, 2025.xxxx; doi: [xxxx](https://doi.org/xxxx)


## Installation
```
pip install -r requirements.txt
```
Note that [CPGNetworkSimulator](https://github.com/SimonDanner/CPGNetworkSimulator) requires boost headers, so make sure they are in the search path.

## Usage
### Run simulations with noise:
```
python simulate_noise.py -y [path to simulation yaml file]
```
### Bifurcation diagrams
Use Jupyter notebook `bifurcation_diagrams.ipynb`. Can also be run on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab-link_bifurcations] without installation.

### Data analysis and figures
Use Jupyter notebook `analyze_results.ipynb`. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab-link_analysis]

## Files

### Data

- `data/df_raw.h5`: Raw rat data as table with meta data (from [Danner et. al 2023][danner-2023-link])

- `data/df_phases.h5`: Table of processed rat data, includes phase differences and other locomotion parameters as well as gait classifcation (from [Danner et. al 2023][danner-2023-link])

- `data/df_intact_sim_8s_final.h5`: Saved simulation results of pre-injury/intact model

- `data/df_hemi_sim_8s_final.h5`: Saved simulation results of hemisection injury case

- `data/df_contusion_sim_8s_final.h5`: Saved simulation results of contusion injury case

### Model files

- `models/RM_final.txt`: Neural network configuration file (for [CPGNetworkSimulator][https://github.com/SimonDanner/CPGNetworkSimulator]) of pre-injury model 

- `models/*.yaml`: Simulation configuration files for the various cases [intact/pre-injury, hemisection (including partial manipulations), contusion]. These files specify simulation paramters and patch connection weights and drive parameters of the model config file (RM_final.txt) to simulate injury conditions.

### Code

- `yaml_sim.py`: Helper functions to read yaml files and update/modify network parameters. 

- `classify_gaits.py`: Calculates gait classifiction based on phase differences and duty factors

- `simulate_noise.py`: Main simulation script t

- `analyze_results.ipynb`: Jupyter notebook containing data analysis and presentation of both modelling results and experimental data. Creates most plots included in the paper. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab-link_analysis]

- `bifurcation_diagrams.ipynb`: Jupyter notebook containing simulation code to run bifurcaton diagrams presented in the paper. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab-link_bifurcations]


## Licenses
Data licensed under Creative Commons Attribution-ShareAlike 4.0 International License [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

Source code licensed under GNU GPL v3 [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)][gpl3]

## References
Danner SM, Shepard CT, Hainline C, Shevtosva NA, Rybak IA, Magnuson DSK. (2023). **Spinal control of locomotion before and after spinal cord injury**. *Experimental Neurology, 368*, 114496. doi: [10.1016/j.expneurol.2023.114496](https://doi.org/10.1016/j.expneurol.2023.114496). Preprint: *bioRxiv*, 2023.03.22.533794; doi: [10.1101/2023.03.22.533794](https://doi.org/10.1101/2023.03.22.533794)

[gpl3]: https://www.gnu.org/licenses/gpl-3.0
[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


[danner-2023-link]: https://doi.org/10.1016/j.expneurol.2023.114496
[colab-link_analysis]: https://colab.research.google.com/github/dannerlab/rat-sci-locomotion-model/blob/master/analyze_results.ipynb
[colab-link_bifurcations]: https://colab.research.google.com/github/dannerlab/rat-sci-locomotion-model/blob/master/bifurcation_diagrams.ipynb
[binder-link]: https://mybinder.org/v2/gh/dannerlab/rat-sci-locomotion/HEAD?labpath=Analysis.ipynb