rm *.h5
mpirun --allow-run-as-root -np 4 python train_cluster.py
