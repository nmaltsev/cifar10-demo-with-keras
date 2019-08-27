# cifar10-demo-with-keras
This research was done to demonstrate distributed CNN training on a cluster.
The target platforms: Jetson TK1, x86 CPU.


1. To prepare dataset: `python dataset.py`
2. To train the network `python train.py`
3. To run distributed training on the cluster (one host) `./cluster_emulated.sh`
4. To run distributed training on the cluster `./cluster_distributed.sh`


### Configuration for MPI

There can be some issues if you try to execute code in docker.

```
apt-get remove python-mpi4py
apt-get install libopenmpi-dev
pip install mpi4py

```
