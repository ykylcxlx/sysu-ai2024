import argparse
import gym
from argument import dqn_arguments#, pg_arguments
import matplotlib.pyplot as plt
import json

def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    # parser.add_argument('--train_dqn', default=False, type=bool, help='whether train DQN')
    parser.add_argument("--train_dqn", action="store_true", help="是否运行dqn模型训练")
    parser.add_argument("--device", type=str, default=None, help="训练时候加载的设备: cpu/gpu, e.g. cuda:0")
    parser = dqn_arguments(parser)
    #parser = pg_arguments(parser)
    args = parser.parse_args()
    print(args)
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_dqn_after import AgentDQN
        agent = AgentDQN(env, args)      
        agent.run()
def draw():
    with open('output/reward.json', 'r') as f:
        data = json.load(f)

    x_axis = list(range(len(data['train'])))
    train_scores = data['train']
    eval_scores = data['eval']

    plt.figure(figsize=(10, 5))  # 设置图像大小
    plt.plot(x_axis, train_scores, label='Training Score', marker='o',markersize=3)  # 绘制训练分数
    plt.plot(x_axis, eval_scores, label='Evaluation Score',marker='*')  # 绘制评估分数

    # 添加图例和标题
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training and Evaluation Scores Over Epochs')
    plt.legend()  # 显示图例

    # 展示图表
    plt.show()

if __name__ == '__main__':
    args = parse()
    run(args)
    draw()
