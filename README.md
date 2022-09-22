# DigitalTwin
**[UNDER CONSTRUCTION]**
DenseFusion Lightning Module, reference: https://github.com/j96w/DenseFusion

OpenARK 3D pose estimation codebase.

## Set Up
```
conda install python=3.8
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge tensorboard
conda install -c anaconda scipy 
conda install -c open3d-admin open3ds
pip install pytorch-lightning
cd pointnet
python setup.py install
```

## Train
```
python3 trainer.py
```

## Visualize Tensorboard Logs
```
tensorboard --logdir tb_logs/dense_fusion/version_0
```

Replace `0` with whichever version you are running.
