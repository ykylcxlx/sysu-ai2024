def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')
    
    parser.add_argument("--hidden_size", default=128, type=int, help="Q-Net隐藏层大小")
    parser.add_argument("--lr", default=0.01, type=float, help="学习率")
    parser.add_argument("--gamma", default=0.9, type=float, help="gamma参数，损失计算时使用")
    parser.add_argument("--epsilon", default=0.1, type=float, help="epsilon参数，ReplayBuffer中使用")
    parser.add_argument("--batch_size", default=100, type=int, help="训练批次大小")
    parser.add_argument("--epoch", default=250, type=int, help="训练轮数")

    parser.add_argument("--saved_dir", default="output", type=str, help="结果保存文件夹")

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser
