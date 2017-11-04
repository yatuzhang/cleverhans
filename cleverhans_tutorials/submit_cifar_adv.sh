srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --crafting_eps 0.1
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --crafting_eps 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --crafting_eps 0.5
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --crafting_eps 0.7

srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.1 --crafting_eps 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --crafting_eps 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.5 --crafting_eps 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.7 --crafting_eps 0.3

srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --lmbda 0.1
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --lmbda 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --lmbda 0.5
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --lmbda 0.7

srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.1 --lmbda 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.3 --lmbda 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.5 --lmbda 0.3
srun -p gpuc --gres=gpu:1 python mnist_blackbox_cifar_adv.py --training_eps 0.7 --lmbda 0.3