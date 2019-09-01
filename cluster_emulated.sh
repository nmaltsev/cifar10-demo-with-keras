number_of_workers=4
# dataset_path="/media/cluster_files/dev/cifar/cifar-10-batches-py"
dataset_path="/root/tfplayground/datasets/cifar-10-batches-py"

mkdir -p data
python dataset.py $dataset_path $number_of_workers

### rm *.h5
mpirun --allow-run-as-root -np 4 python train_cluster.py
