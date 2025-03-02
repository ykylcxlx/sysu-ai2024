import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
import logging
import codecs
import json

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        hidden_size = 64
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, inputs):
        return self.fc(inputs)
        


class ReplayBuffer:
    # 环境数据缓存
    def __init__(self, buffer_size: int, n_states: int):
        self.max_buffer_size = buffer_size # 最大的缓存空间
        self.buffer = np.zeros((buffer_size, n_states * 2 + 3)) # 缓存大小: [max_buffer_size, n_states * 2 + 3]
        self.buffer_counter = 0 # 缓存空间更新计数
        

    def __len__(self):
        return self.buffer_counter

    def push(self, *transition):
        """新增环境数据
        """
        
        # transition[state, action, reward, terminated, next_state]
        s = transition[0] # state: dim[4]
        a = transition[1] # action: int
        r = transition[2] # reward: float
        t = transition[3] # terminated: int
        s_ = transition[4] # next_state: dim[4]
        
        transition = np.hstack((s, [a, r, t], s_))
        
        index = self.buffer_counter % self.max_buffer_size
        
        self.buffer[index, :] = transition
        
        self.buffer_counter += 1
        if self.buffer_counter % 100 == 0:
            logging.info(f"buffer_counter: {self.buffer_counter}, push_index: {index}")

    def sample(self, batch_size):
        """采样训练数据

        Args:
            batch_size (int): batch size

        Returns:
            np.ndarray: batch训练数据
        """

        sample_index = np.random.choice(self.max_buffer_size, batch_size)
        batch_buffer = self.buffer[sample_index, :]
        return batch_buffer

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        pass


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)

        self.env = env
        self.args = args
        self.epsilon = self.args.epsilon
        self.batch_size = self.args.batch_size

        self.init_game_setting()
    

    def init_game_setting(self):
        """初始化配置
        """
        self.n_states = self.env.observation_space.shape[0] # 4
        self.n_actions = self.env.action_space.n # 2
        
        self.learn_step_counter = 0
        self.buffer = ReplayBuffer(buffer_size=2000, n_states=self.n_states)

        self.q_net = QNetwork(input_size=self.n_states, 
                              hidden_size=self.args.hidden_size,
                              output_size=self.n_actions)
        self.q_net.to(self.args.device)
        self.target_net = QNetwork(input_size=self.n_states, 
                                   hidden_size=self.args.hidden_size,
                                   output_size=self.n_actions)
        self.target_net.to(self.args.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()

    def train(self):
        """网络训练
        """
        self.q_net.train()
        self.target_net.eval()

        # 获取训练数据，以及预处理
        b_memory = self.buffer.sample(batch_size=self.batch_size)
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(self.args.device) # 当前状态
        print(b_s.shape)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)) .to(self.args.device)# 动作
        print(b_a.shape)
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(self.args.device) # 收益
        print(b_r.shape)
        b_t = torch.LongTensor(b_memory[:, self.n_states+2:self.n_states+3]).to(self.args.device) # 结束标志
        print(b_t.shape)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(self.args.device) # 下一步状态
        print(b_s_.shape)
        
        # 网络计算
        q = self.q_net(b_s).gather(1, b_a) # [batch, 1], 获取当前状态下对应b_a动作位置的q_value
        q_next = self.target_net(b_s_).detach().max(1)[0] # [batch]
        q_next = q_next.view(self.batch_size, 1) # [batch, 1]
        # q_target = b_r + self.args.gamma * q_next * (1 - b_t) # [batch, 1], truncated=1的话,target=reward;否则target=reward+gamma*next
        q_target = b_r + self.args.gamma * q_next # [batch, 1]

        # 损失计算
        loss = self.loss_func(q, q_target)

        # 梯度回传
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_action(self, observation, test=False):
        """预测下一步动作
        """
        # observation 是一个状态: dim[4]
        x = torch.unsqueeze(torch.FloatTensor(observation), 0)

        if test or (not test and np.random.uniform() < self.epsilon):
            # greedy policy
            self.q_net.eval()
            with torch.no_grad():
                actions_value = self.q_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            # random
            action = np.random.randint(0, self.n_actions)
        return action

    def run(self):
            """Implement the interaction between agent and environment here
            """

            sample_step_counter = 0 # 用来计算采样步数
            train_step_counter = 0 # 用来计算训练步数

            train_reward_list = []
            eval_reward_list = []

            for i_epoch in range(self.args.epoch):
                
                s, _ = self.env.reset() # dim [4]
                total_reward = 0

                while True:
                    sample_step_counter += 1

                    self.env.render()
                    a = self.make_action(s) # 0 / 1

                    # 采取动作
                    # terminated: Pole Angle is greater than ±12°, Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
                    # truncated: Episode length is greater than 500 (200 for v0)
                    s_, r, terminated, truncated, _ = self.env.step(a)

                    # 重新定义reward, reward越大越好
                    x, x_dot, theta, theta_dot = s_ # x: 小车绝对位置, theta: 木棍的倾斜角度
                    r1_ = (self.env.x_threshold - abs(x)) / self.env.x_threshold
                    r2_ = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians
                    r_ = r1_ + r2_

                    total_reward += r # 累加收益
                    
                    self.buffer.push(s, a, r_, int(terminated), s_) # 记录样本
                    #print(type(self.buffer))
                    if sample_step_counter > 10 and len(self.buffer) > self.buffer.max_buffer_size:
                        #print(len(self.buffer),self.buffer.max_buffer_size)
                        #exit(0)
                        
                        sample_step_counter = 0 # 新增10条数据，采样训练一次
                        self.epsilon = min(self.epsilon * 1.01 , 1) # 更新epsilon
                        self.train()
                        train_step_counter += 1
                        if train_step_counter % 20 == 0:
                            # 每训练20轮，同步一次参数到target_net
                            self.target_net.load_state_dict(self.q_net.state_dict())
                            logging.info(f">>>>>>>> update target_net.")                    
                    
                    if terminated or truncated:
                        break

                    s = s_
                
                if train_step_counter > 0:
                    # 每次epoch结束输出一次测试结果
                    eval_reward = self.eval()
                    eval_reward_list.append(eval_reward)
                    train_reward_list.append(total_reward)
                    logging.info(f"epoch: {i_epoch}, eval: {eval_reward}, train: {total_reward}, epsilon: {self.epsilon}")

            # 保存模型
            
            saved_model_path = os.path.join(self.args.saved_dir, "model.pt")
            os.makedirs(os.path.dirname("./"+saved_model_path), exist_ok=True)
            print(f"save model to {saved_model_path}")
            torch.save(self.q_net.state_dict(), saved_model_path)

            # 保存训练过程
            saved_reward_path = os.path.join(self.args.saved_dir, "reward.json")
            reward_trace = {
                "train": train_reward_list,
                "eval": eval_reward_list
            }
            with codecs.open(saved_reward_path, "w", "utf-8") as f:
                json.dump(reward_trace, f, indent=4)
    def eval(self):
        """eval
        """
        self.q_net.eval()

        s, _ = self.env.reset()
        total_reward = 0

        while True:
            a = self.make_action(s, test=True)
            s_, r, terminated, truncated, _ = self.env.step(a)
            total_reward += r
            s = s_

            if terminated or truncated:
                break
        
        return total_reward
