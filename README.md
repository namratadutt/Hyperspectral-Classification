# Hyperspectral-Classification
### Classification of different landcover classes using Hyperspectral data.

<hr>

#### Steps to Run:

1. Create a virtual environment using command: ```virtualenv myenv```
2. Activate the virtual environment: ```source venv/bin/activate ```
3. Install the requirements file: ```pip install -r requirements.txt```
4. Download [this](https://github.com/GatorSense/MUUFLGulfport/blob/master/MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat) gulfport mat file in the same directory.
5. Run the file: ```python main.py```

<hr>

#### Results:
An accuracy of 87.98% ± 0.71 was achieved with Fully connected neural network. The Confusion matrix is shown below:
<img width="887" alt="hsi_github" src="https://user-images.githubusercontent.com/50796784/173253012-bd1ae18f-9f2f-4d01-940b-36978a4a1265.png">


#### References

P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, “MUUFL Gulfport Hyperspectral and LiDAR Airborne Data Set,” University of Florida, Gainesville, FL, Tech. Rep. REP-2013-570, Oct. 2013.

X. Du and A. Zare, “Technical Report: Scene Label Ground Truth Map for MUUFL Gulfport Data Set,” University of Florida, Gainesville, FL, Tech. Rep. 20170417, Apr. 2017.

<hr>
