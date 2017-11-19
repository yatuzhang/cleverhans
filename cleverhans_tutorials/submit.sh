srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_tutorial_tf.py --eps 0.1
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_tutorial_tf.py --eps 0.3
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_tutorial_tf.py --eps 0.5
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_tutorial_tf.py --eps 0.7
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_tutorial_tf.py --eps 0.9

srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_gpdnn_tf.py --eps 0.1
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_gpdnn_tf.py --eps 0.3
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_gpdnn_tf.py --eps 0.5
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_gpdnn_tf.py --eps 0.7
srun -p gpuc --gres=gpu:1 -w guppy16 python mnist_gpdnn_tf.py --eps 0.9