# MultiBarrelModel
![fig](MultiBarrelModel_outline.png "Magic Gardens")
Code implementation of our paper ['Localist Topographic Expert Routing: A Barrel Cortex-Inspired Modular Network for Sensorimotor Processing'](https://neurips.cc/virtual/2025/poster/120226) (NeurIPS 2025). 
## Dependency
Core dependencies: Python 3.10 and PyTorch 1.12.1. See requirements.txt for additional packages. Run
```bash
pip install -r requirements.txt
```
## Datasets
This work primarily uses the ```EvTouch-Objects``` and ```EvTouch-Containers``` tactile datasets, consisting of 36 and 20 classes respectively. For detailed information about these datasets, please refer to [TactileSGNet](https://github.com/clear-nus/TactileSGNet).
## File description
* ```/data/:``` Contains raw files for both tactile datasets (EvTouch-Objects and EvTouch-Containers).
* ```dataset.py:``` Loads two tactile datasets.
* ```MultiBarrel4EvTask.py:``` Train a multi-barrel model with 39 independently parameterized barrels.
* ```SharedMultiBarrel4EvTask.py:``` Train a multi-barrel model with 39 barrels sharing training parameters.
* ```SingleBarrel4EvTask.py:``` Train a single-barrel model with neuron count matching the above two models.
* ```utils.py:``` Some auxiliary modules in the model (e.g., single-neuron dynamics).
* ```MultiBarrel_simulate.py:``` Simulate optogenetic experiments to observe the spread of neural activity.
* ```Model_lossLandscape.py:``` Visualize the loss landscape of models.
* ```MultiBarrel_propagation.py:``` Measure neural activity correlation between barrels.
## Train models
The code is almost one-click runnable. Once the dataset files are correctly placed in the ```./data/``` directory, you can train either shared-parameter or independent-parameter multi-barrel models by executing the corresponding ```.py``` file directly. For example:
```python
python SharedMultiBarrel4EvTask.py
```
Note that the ```EvTouch-Objects``` and ```EvTouch-Containers``` datasets contain different numbers of classes.
## Simulate optogenetic experiments
Similarly, run ```MultiBarrel_simulate.py``` to visualize the temporal spread of neural activity:
```python
python MultiBarrel_simulate.py
```
## Loss landscape 
Load the trained model weights to visualize the loss landscape:
```python
python Model_lossLandscape.py
```
## Measure neural activity correlation
Load the trained model weights to calculate both global and local neural activity correlations across barrel pairs:
```python
python MulitBarrel_Propagation.py
```
## Citation
If you find this work useful, please cite:
**Please raise any questions related to the code.**
