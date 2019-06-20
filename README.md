# cifar10-demo-with-keras
Implementation for Jetson TK1


1. Prepare dataset: `python dataset.py`
2. Train the network `python train.py`


### Configuration for MPI

There can be some issues if you try to execute code in docker.

```
apt-get remove python-mpi4py
apt-get install libopenmpi-dev
pip install mpi4py

```
