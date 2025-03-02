import argparse 
from argument import dqn_arguments, pg_arguments
parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
# parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
# parser.add_argument('--train_dqn', default=False, type=bool, help='whether train DQN')
parser.add_argument("--train_dqn", action="store_true", help="是否运行dqn模型训练")
parser.add_argument("--device", type=str, default=None, help="训练时候加载的设备: cpu/gpu, e.g. cuda:0")
# parser = dqn_arguments(parser)
parser = pg_arguments(parser)
#opts = parser.parse_args()
print(parser.parse_args())
