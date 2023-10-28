"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).

通过修改gym环境，通过A3C方式实现迷宫寻路
"""
import datetime
import random
import sys
import time

import torch
import torch.nn as nn
import xlsxwriter

from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.95
MAX_EP = 10001
SAVE_MODEL_EVERY = 5
LOAD_MODEL = False

game = 'NetworkWorld-v0'
env = gym.make(game)  # GridWorld-v0 CartPole-v0
N_S = env.observation_space.shape[0]  # 查看这个环境中observation的特征有多少个，返回int 4
# print(N_S)
N_A = env.action_space.n  # 查看这个环境中可用的action有多少个，返回int 2 左 右
TRAIN_OVER = 0


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        with open('tmp.txt', 'w') as file:
            file.write('')

        # print("s_dim",s_dim)
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)  # 状态全连接层
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        # prob = F.softmax(logits, dim=1).data #将张量的每个元素缩放到（0,1）区间且和为1
        prob = F.softmax(logits, dim=0).data  # 将张量的每个元素缩放到（0,1）区间且和为1
        m = self.distribution(prob)  # 按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引。
        return m.sample().numpy()

    def choose_action_random(self, ):
        self.eval()
        prob = torch.rand(N_A)
        m = self.distribution(prob)  # 按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引。
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, block_queue, name,
                 output_block_queue, output_block_due2_overtime_queue, output_block_due2_reconfig_queue,
                 output_block_due2_memory_queue, output_block_due2_opt_only_queue, output_block_due2_IP_only_queue,
                 output_block_due2_optIP_queue,
                 output_delay_queue, output_delay_due2_reconfig_queue, output_delay_forward_queue,
                 output_delay_propagation_queue, output_delay_process_queue,
                 output_opt_util_queue, output_ip_util_queue, output_memory_util_queue,
                 output_optical_reconfig_times_queue, output_ip_reconfig_times_queue,
                 output_comput_reconfig_times_queue):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.reward_queue, self.block_queue = global_ep, global_ep_r, res_queue, block_queue

        self.output_block_queue, self.output_block_due2_overtime_queue, self.output_block_due2_reconfig_queue \
            = output_block_queue, output_block_due2_overtime_queue, output_block_due2_reconfig_queue
        self.output_block_due2_memory_queue, self.output_block_due2_opt_only_queue, self.output_block_due2_IP_only_queue, self.output_block_due2_optIP_queue \
            = output_block_due2_memory_queue, output_block_due2_opt_only_queue, output_block_due2_IP_only_queue, output_block_due2_optIP_queue
        self.output_delay_queue, self.output_delay_due2_reconfig_queue, self.output_delay_forward_queue, self.output_delay_propagation_queue, self.output_delay_process_queue \
            = output_delay_queue, output_delay_due2_reconfig_queue, output_delay_forward_queue, output_delay_propagation_queue, output_delay_process_queue
        self.output_opt_util_queue, self.output_ip_util_queue, self.output_memory_util_queue \
            = output_opt_util_queue, output_ip_util_queue, output_memory_util_queue
        self.optical_reconfig_times_queue, self.ip_reconfig_times_queue, self.comput_reconfig_times_queue \
            = output_optical_reconfig_times_queue, output_ip_reconfig_times_queue, output_comput_reconfig_times_queue

        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = gym.make(game).unwrapped  # 打开限制

    def run(self):
        global LOAD_MODEL
        global TRAIN_OVER
        total_step = 1
        done_time = 0

        while self.g_ep.value < MAX_EP:

            if LOAD_MODEL:
                LOAD_MODEL = False
                check_point = torch.load(str(self.name) + '.tar')
                self.lnet.load_state_dict(check_point['model_state_dict'])
                self.opt.load_state_dict(check_point['optimizer_state_dict'])
                done_time = check_point['done_time']

                print(self.name, done_time)
                # 保存self.res_queue和block qeueu
                '''
                self.res_queue = check_point['res_queue']
                self.block_queue = check_point['block_queue']
                '''
            s = self.env.reset()  # 重置
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            ep = 0
            while True:
                # if self.name == 'w00':
                #    self.env.render() #重置环境 图像引擎
                # a = self.lnet.choose_action(v_wrap(s[None, :])) #选择动作 原本
                if ep < 1:
                    a = self.lnet.choose_action_random()  # 随机选择动作
                else:
                    a = self.lnet.choose_action(v_wrap(s))
                ep += 1
                # a = self.lnet.choose_action(v_wrap(s))

                s_, r, done, block_rate, \
                output_block, output_block_due2_overtime, output_block_due2_reconfig, output_block_due2_memory, output_block_due2_opt_only, output_block_due2_IP_only, output_block_due2_optIP, \
                output_delay, output_delay_due2_reconfig, output_delay_forward, output_delay_propagation, output_delay_process, \
                output_opt_util, output_ip_util, output_memory_util, optical_reconfig_times, ip_reconfig_times, comput_reconfig_times \
                    = self.env.step(a)  # 根据动作返回下一个状态，回报，是否结束
                # if done: r = -1 #如果为结束状态，r = -1
                ep_r += r  # 总回报r+ = r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # 如果本轮学习结束或结束状态 update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information

                        record(self.g_ep, self.g_ep_r, ep_r, self.reward_queue, self.name, block_rate, self.block_queue,
                               output_block, self.output_block_queue,
                               output_block_due2_overtime, self.output_block_due2_overtime_queue,
                               output_block_due2_reconfig, self.output_block_due2_reconfig_queue,
                               output_block_due2_memory, self.output_block_due2_memory_queue,
                               output_block_due2_opt_only, self.output_block_due2_opt_only_queue,
                               output_block_due2_IP_only, self.output_block_due2_IP_only_queue,
                               output_block_due2_optIP, self.output_block_due2_optIP_queue,
                               output_delay, self.output_delay_queue,
                               output_delay_due2_reconfig, self.output_delay_due2_reconfig_queue,
                               output_delay_forward, self.output_delay_forward_queue,
                               output_delay_propagation, self.output_delay_propagation_queue,
                               output_delay_process, self.output_delay_process_queue,
                               output_opt_util, self.output_opt_util_queue,
                               output_ip_util, self.output_ip_util_queue,
                               output_memory_util, self.output_memory_util_queue,
                               optical_reconfig_times, self.optical_reconfig_times_queue,
                               ip_reconfig_times, self.ip_reconfig_times_queue,
                               comput_reconfig_times, self.comput_reconfig_times_queue,

                               )

                        done_time += 1

                        # Save model every: self.save_model_every
                        if done_time % SAVE_MODEL_EVERY == 0:
                            torch.save({
                                'model_state_dict': self.lnet.state_dict(),
                                'optimizer_state_dict': self.opt.state_dict(),
                                'done_time': done_time,
                            }, str(self.name) + '.tar')
                            # print(self.name, "MODEL SAVED", done_time)

                        break
                s = s_
                total_step += 1
        #         print("in true loop")
        #     print("in MAX_EP loop")

        # print("self.g_ep.value, MAX_EP", self.g_ep.value, MAX_EP)
        #
        # print("change_train_over()", change_train_over())
        # print("get_train_over()", get_train_over())
        # self.reward_queue.put(None)
        # self.block_queue.put(None)
        # self.output_block_queue.put(None)
        # self.output_block_due2_overtime_queue.put(None)
        # self.output_block_due2_reconfig_queue.put(None)
        # self.output_block_due2_memory_queue.put(None)
        # self.output_block_due2_opt_only_queue.put(None)
        # self.output_block_due2_IP_only_queue.put(None)
        # self.output_block_due2_optIP_queue.put(None)
        # self.output_delay_queue.put(None)
        # self.output_delay_due2_reconfig_queue.put(None)
        # self.output_delay_forward_queue.put(None)
        # self.output_delay_propagation_queue.put(None)
        # self.output_delay_process_queue.put(None)
        # self.output_opt_util_queue.put(None)
        # self.output_ip_util_queue.put(None)
        # self.output_memory_util_queue.put(None)
        # self.optical_reconfig_times_queue.put(None)
        # self.ip_reconfig_times_queue.put(None)
        # self.comput_reconfig_times_queue.put(None)
        # print("run over", self.name)


def queue2list(queue):
    l = []
    while True:
        r = queue.get()
        if queue.qsize() != 0:
            l.append(r)
        else:
            break
    return l


def output_result_during_training_in_multi_col(dic, file_name):  # 将结果写入文件
    workbook = xlsxwriter.Workbook(file_name + '.xlsx')  # 创建一个Excel文件
    worksheet = workbook.add_worksheet()  # 创建一个sheet
    col = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
           "W", "X", "Y", "Z"]
    i = 0
    for key in dic.keys():
        title = [key]  # 表格title
        worksheet.write_row(col[i] + "1", title)  # title 写入Excel
        tmp = dic.get(key)
        l = []
        for m in range(len(tmp)):
            if m % 1 == 0:
                l.append(tmp[m])
        for j in range(len(l)):
            num0 = j + 2
            row = col[i] + str(num0)
            d = [str(l[j])]
            worksheet.write_row(row, d)
            # i += 1
        i += 1
    workbook.close()


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer

    global_ep, global_ep_r, reward_queue, block_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Queue()
    output_block_queue, output_block_due2_overtime_queue, output_block_due2_reconfig_queue, output_block_due2_memory_queue = mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue()
    output_block_due2_opt_only_queue, output_block_due2_IP_only_queue, output_block_due2_optIP_queue = mp.Queue(), mp.Queue(), mp.Queue()
    output_delay_queue, output_delay_due2_reconfig_queue, output_delay_forward_queue, output_delay_propagation_queue, output_delay_process_queue \
        = mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue()
    output_opt_util_queue, output_ip_util_queue, output_memory_util_queue = mp.Queue(), mp.Queue(), mp.Queue()
    output_optical_reconfig_times_queue, output_ip_reconfig_times_queue, output_comput_reconfig_times_queue = mp.Queue(), mp.Queue(), mp.Queue()
    # parallel training
    workers = [
        Worker(
            gnet, opt, global_ep, global_ep_r, reward_queue, block_queue, i,
            output_block_queue, output_block_due2_overtime_queue, output_block_due2_reconfig_queue,
            output_block_due2_memory_queue,
            output_block_due2_opt_only_queue, output_block_due2_IP_only_queue, output_block_due2_optIP_queue,
            output_delay_queue, output_delay_due2_reconfig_queue, output_delay_forward_queue,
            output_delay_propagation_queue, output_delay_process_queue,
            output_opt_util_queue, output_ip_util_queue, output_memory_util_queue,
            output_optical_reconfig_times_queue, output_ip_reconfig_times_queue, output_comput_reconfig_times_queue
        )
        for i in range(mp.cpu_count() - 8)
        # for i in range(1)
    ]
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(1)]
    [w.start() for w in workers]

    # [w.terminate() for w in workers]

    # [w.join() for w in workers]
    # [w.is_alive() for w in workers]
    while reward_queue.qsize() < MAX_EP - 1:
        time.sleep(5)
    end_time = datetime.datetime.now()
    print("start time", start_time, "end time", end_time, "time", end_time - start_time)
    print("log转list")
    reward = queue2list(reward_queue)  # record episode reward to plot
    # 画图
    print("log画图")
    plt.figure("block")
    # plt.subplot(421)
    plt.plot(reward)
    plt.ylabel('reward')
    plt.xlabel('Step')
    plt.show()

    # 存储结果数据
    '''
    while reward_queue.qsize() < MAX_EP - 1:
        time.sleep(5)
    # for w in workers:
    #     print(w.name)
    #     w.join()
    #     print("in loop len(workers)", len(workers))

    print("log转list")
    reward = queue2list(reward_queue)                    # record episode reward to plot
    output_block = queue2list(output_block_queue)
    output_block_due2_overtime = queue2list(output_block_due2_overtime_queue)
    output_block_due2_reconfig = queue2list(output_block_due2_reconfig_queue)
    output_block_due2_memory = queue2list(output_block_due2_memory_queue)
    output_block_due2_opt_only = queue2list(output_block_due2_opt_only_queue)
    output_block_due2_IP_only = queue2list(output_block_due2_IP_only_queue)
    output_block_due2_optIP = queue2list(output_block_due2_optIP_queue)
    output_delay = queue2list(output_delay_queue)
    output_delay_due2_reconfig = queue2list(output_delay_due2_reconfig_queue)
    output_delay_forward = queue2list(output_delay_forward_queue)
    output_delay_propagation = queue2list(output_delay_propagation_queue)
    output_delay_process = queue2list(output_delay_process_queue)
    output_opt_util = queue2list(output_opt_util_queue)
    output_ip_util = queue2list(output_ip_util_queue)
    output_memory_util = queue2list(output_memory_util_queue)
    output_optical_reconfig_times = queue2list(output_optical_reconfig_times_queue)
    output_ip_reconfig_times = queue2list(output_ip_reconfig_times_queue)
    output_comput_reconfig_times = queue2list(output_comput_reconfig_times_queue)
    block = queue2list(block_queue)                    # record episode block rate to plot




    print("log存储结果数据")
    print("len比较",len(reward), len(block), len(output_ip_util) )
    result = {}
    result["reward"] = reward
    # result[block] = block
    result["output_block"] = output_block
    result["output_block_due2_overtime"] = output_block_due2_overtime
    result["output_block_due2_reconfig"] = output_block_due2_reconfig
    result["output_block_due2_memory"] = output_block_due2_memory
    result["output_block_due2_opt_only"] = output_block_due2_opt_only
    result["output_block_due2_IP_only"] = output_block_due2_IP_only
    result["output_block_due2_optIP"] = output_block_due2_optIP
    result["output_delay"] = output_delay
    result["output_delay_due2_reconfig"] = output_delay_due2_reconfig
    result["output_delay_forward"] = output_delay_forward
    result["output_delay_propagation"] = output_delay_propagation
    result["output_delay_process"] = output_delay_process
    result["output_opt_util"] = output_opt_util
    result["output_ip_util"] = output_ip_util
    result["output_memory_util"] = output_memory_util

    result["output_optical_reconfig_times"] = output_optical_reconfig_times
    result["output_ip_reconfig_times"] = output_ip_reconfig_times
    result["output_comput_reconfig_times"] = output_comput_reconfig_times

    output_result_during_training_in_multi_col(result, "result")
    # 画图
    print("log画图")
    plt.figure("block")
    plt.subplot(421)
    plt.plot(reward)
    plt.ylabel('reward')
    plt.xlabel('Step')

    plt.subplot(422)
    plt.plot(block)
    plt.ylabel('block')
    plt.xlabel('Step')

    plt.subplot(423)
    plt.plot(output_block_due2_overtime)
    plt.ylabel('output_block_due2_overtime')
    plt.xlabel('Step')

    plt.subplot(424)
    plt.plot(output_block_due2_reconfig)
    plt.ylabel('output_block_due2_reconfig')
    plt.xlabel('Step')

    plt.subplot(425)
    plt.plot(output_block_due2_memory)
    plt.ylabel('output_block_due2_memory')
    plt.xlabel('Step')

    plt.subplot(426)
    plt.plot(output_block_due2_opt_only)
    plt.ylabel('output_block_due2_opt_only')
    plt.xlabel('Step')

    plt.subplot(427)
    plt.plot(output_block_due2_IP_only)
    plt.ylabel('output_block_due2_IP_only')
    plt.xlabel('Step')

    plt.subplot(428)
    plt.plot(output_block_due2_optIP)
    plt.ylabel('output_block_due2_optIP')
    plt.xlabel('Step')

    plt.figure("delay")
    plt.subplot(321)
    plt.plot(output_delay)
    plt.ylabel('output_delay')
    plt.xlabel('Step')

    plt.subplot(322)
    plt.plot(output_delay_due2_reconfig)
    plt.ylabel('output_delay_due2_reconfig')
    plt.xlabel('Step')

    plt.subplot(323)
    plt.plot(output_delay_forward)
    plt.ylabel('output_delay_forward')
    plt.xlabel('Step')

    plt.subplot(324)
    plt.plot(output_delay_propagation)
    plt.ylabel('output_delay_propagation')
    plt.xlabel('Step')

    plt.subplot(325)
    plt.plot(output_delay_process)
    plt.ylabel('output_delay_process')
    plt.xlabel('Step')

    plt.figure("resource")
    plt.subplot(311)
    plt.plot(output_opt_util)
    plt.ylabel('output_opt_util')
    plt.xlabel('Step')

    plt.subplot(312)
    plt.plot(output_ip_util)
    plt.ylabel('output_ip_util')
    plt.xlabel('Step')

    plt.subplot(313)
    plt.plot(output_memory_util)
    plt.ylabel('output_memory_util')
    plt.xlabel('Step')

    plt.figure("reconfig times")
    plt.subplot(311)
    plt.plot(output_optical_reconfig_times)
    plt.ylabel('output_optical_reconfig_times')
    plt.xlabel('Step')

    plt.subplot(312)
    plt.plot(output_ip_reconfig_times)
    plt.ylabel('output_ip_reconfig_times')
    plt.xlabel('Step')

    plt.subplot(313)
    plt.plot(output_comput_reconfig_times)
    plt.ylabel('output_comput_reconfig_times')
    plt.xlabel('Step')

    print("log开始show图")
    plt.show()
    print("log结束show图")

    print("log在w.join() for w in workers前")
'''
