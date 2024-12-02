# ImDy: Human Inverse Dynamics from Imitated Observations

The code release is in progress.

## Environment Setup

Create a conda environment from `environment.yml`: `conda env create -f environment.yml`

## Data Acquisition

For data access, please get in touch with xinpengliu0907@gmail.com. 

The file structure should be 

```
- utils
- data
 |- raw_test
   |- grf.pkl
   |- pos.pkl
   |- rot.pkl
   |- torque.pkl
   |- weight.pkl
 |- raw_train
   |- ...
- models
 |- containing SMPLX models from https://smpl-x.is.tue.mpg.de/
- convert.py
```

Run ``python convert.py`` to convert the raw data into a different format with per-sample pickle files including axis-angle format SMPL parameters, joints, and markers. 
The torques stored are acquired by summing two consecutive torques in the simulation. 
