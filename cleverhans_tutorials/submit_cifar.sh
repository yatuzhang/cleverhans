srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --eps 0.1
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --eps 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --eps 0.5
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --eps 0.7
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --eps 0.9

srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --nb_epochs_s 10
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --nb_epochs_s 20
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --nb_epochs_s 30
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --nb_epochs_s 40
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --nb_epochs_s 50

srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --data_aug 4
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --data_aug 6
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --data_aug 8

srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --lmbda 0.1
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --lmbda 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --lmbda 0.5
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --lmbda 0.7
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar.py --lmbda 0.9
