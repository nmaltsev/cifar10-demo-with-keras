number_of_workers=4
ip_list="./machinefile"
work_dir="/media/cluster_files/dev/repo/cifar10-demo-with-keras/"
dataset_path="/media/cluster_files/dev/cifar/cifar-10-batches-py"

## Attention: There are not enough RAM for preparation dataset chunks! SWAP is required
#~ mkdir -p data
#~ python dataset.py $dataset_path $number_of_workers

## Description of arguments of `mpirun` can be found at https://www.open-mpi.org/faq/?category=running (#18)

#~ mpirun --hostfile ./machinefile --wdir /media/cluster_files/dev/repo/cifar10-demo-with-keras/ -np $number_of_workers python ./train_cluster.py
mpirun --hostfile $ip_list --wdir /media/cluster_files/dev/repo/cifar10-demo-with-keras/ -np $number_of_workers python ./train_cluster.py
