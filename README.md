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
 |- nimble_test
   |- figure
     |- walking
   |- walking.pkl
- osim
 |- Geometry
   |- .....
 |- Rajagopal2015_passiveCal_hipAbdMoved_noArms.osim
 |- vtp_to_ply.py

- models
 |- containing SMPL models from [https://smpl-x.is.tue.mpg.de/](https://smpl.is.tue.mpg.de)
 |- containing Rajagopal2015 model without arm from https://addbiomechanics.org/download_data.html
- convert.py
- adb_motion_visualize.py
```

Run ``python convert.py`` to convert the raw data into a different format with per-sample pickle files including axis-angle format SMPL parameters, joints, and markers. 
The torques stored are acquired by summing two consecutive torques in the simulation. 

Run ``adb_motion_visualize.py`` to visualize the motion from Addbiomechanics Dataset frame by frame.
In line 64, you could change the angles of camera to better visualize the motion.
```
scene.set_camera(angles=(-pi/8,pi/2+pi/4,0),distance=2.5) 
```
![nimble example](./data/nimble_test/figure/walking/0.png)
