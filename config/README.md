# config

`rig_map.json` is the table map, written by `./rig calibrate` and committed: the corners of the table
markers and the table plane in the arm's kinematic base frame, with the fit statistics. It holds no
images. Every camera and arm tool reads it to locate the camera and to know the table height.

Calibrate again with `./rig calibrate --new` and commit the new file whenever the table or a marker
has moved ([docs/real_rig.md](../docs/real_rig.md#calibration)).
