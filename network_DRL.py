from __future__ import division
import logging
import numpy
import random
from gym import spaces
import gym
# import NETWORK_for_DRL
import numpy as np
import time as tm



import functools
from collections import defaultdict
import networkx as nx
import numpy as np
import math

from KSP import k_shortest_paths
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class NetworkEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    TEST_NUM = 1
    RATIO_REQ = 0.5  # 两种业务的比率
    INIT_RATIO = 0.5  # 初始切片比例
    LINK_NUM = 44
    NODE_NUM = 14
    OPTICAL_SLOT_TOTAL = 80
    Ropt_ip = 5  # 光粒度比上IP粒度
    IP_SLOT_TOTAL = 100 * Ropt_ip
    VM_TOTAL = 10  # 每各DC虚拟机数3
    BFS = 12.5 * 60  # 每个频隙带宽 12.5GBPS，一分钟12.5 * 60
    # VM_MEMORY = 500 * 60  # 每个虚拟机内存G
    VM_MEMORY = 500 * 60  # 每个虚拟机内存G
    # vm_process_ability = [[30*0.001 for x in range(MV_TOTAL)] for y in range(NODE_NUM)] # 虚拟机处理速度一分钟多少G
    vm_process_speed = 60 * 60  # 先考虑每个VM处理速度一样 60G每秒

    time_record_req_load = 5  # 多少毫秒记录一次负载
    # num_read_record = 15  # 每次读取的负载数量
    reconfig_frequence = 1  # 每记录reconfig_frequence次进行一次重构
    # delay_increase_per_optical_slice_reconfig_base = 0.5
    # delay_increase_per_ip_slice_reconfig_base = 0.1
    # delay_increase_per_comput_slice_reconfig = 0.1
    ep_A3C = 0 # A3C训练回合数
    delay_increase_per_optical_slice_reconfig = 1.5 # 动态切片增加的时延
    delay_increase_per_ip_slice_reconfig = 0.5 # 动态切片增加的时延
    delay_increase_per_comput_slice_reconfig = 0.2 # 动态切片增加的时延

    NUM_REQ_MEASURE = 20000  # 业务请求总数d
    TMAX = 100  # 若业务阻塞，时延记为TMAX
    DELAY_IN_FIBER = 0.000005/60  # 光纤传输延迟 5 μs/km ，一个时间周期1毫秒
    OVERTIME_TOLERANCE = 10000000  # 时延敏感业务超过Tidea的比例，超过记为阻塞
    OVERTIME_TOLERANCE_for_nonsensitive = 1000000000  # 非时延敏感业务超过Tidea的比例，超过记为阻塞
    over_time_sensitive = 1 # 时延敏感业务超时的比例（使用）

    N = 10  # number of paths each src-dest pair 没用到
    M = 1  # first M starting FS allocation positions are considered 不懂
    kpath = 1  # = 1 SP-FF, = 5, KSP-FF
    LAMBDA_REQ = 15  # 8  每个时间周期内平均请求数 for uniform traffic, = 10, for nonuniform traffic = 16 12
    lambda_time = [1 + 2 * x for x in range(
        11)]  # 单位秒，一个时间周期1毫秒，之后在计算理想时延的时候乘上0.001 average service time per request; randomly select one value from the list for each episode evaluated
    lambda_data = [500 * 0.001]  # 业务数据量，影响业务持续时间 大小影响之后测试一下 放弃使用，根据业务传输时间和理想带宽计算数据量
    bandwidth = [12.5 * 60, 40 * 60]  # G业务带宽需求随机范围
    len_lambda_data = len(lambda_data)
    len_lambda_time = len(lambda_time)

    nonuniform = False  # True#False 均匀业务，随机选起点终点
    OEO_delay_for_delay_sensitive = 0.1
    DC_strategy = "load_balance"  # DC选择策略 random  time_shortest load_balance

    linkmap = defaultdict(lambda: defaultdict(lambda: None))  # Topology: NSFNet
    linkmap[1][2] = (0, 80)  # 编号，长度
    linkmap[2][1] = (3, 80)
    linkmap[1][3] = (1, 80)
    linkmap[3][1] = (6, 80)
    linkmap[1][8] = (2, 80)
    linkmap[8][1] = (22, 80)
    linkmap[2][3] = (4, 80)
    linkmap[3][2] = (7, 80)
    linkmap[2][4] = (5, 80)
    linkmap[4][2] = (9, 80)
    linkmap[3][6] = (8, 80)
    linkmap[6][3] = (15, 80)
    linkmap[4][5] = (10, 80)
    linkmap[5][4] = (12, 80)
    linkmap[4][11] = (11, 80)
    linkmap[11][4] = (32, 80)
    linkmap[5][6] = (13, 80)
    linkmap[6][5] = (16, 80)
    linkmap[5][7] = (14, 80)
    linkmap[7][5] = (19, 80)
    linkmap[6][10] = (17, 80)
    linkmap[10][6] = (29, 80)
    linkmap[6][14] = (18, 80)
    linkmap[14][6] = (41, 80)
    linkmap[7][8] = (20, 80)
    linkmap[8][7] = (23, 80)
    linkmap[7][10] = (21, 80)
    linkmap[10][7] = (30, 80)
    linkmap[8][9] = (24, 80)
    linkmap[9][8] = (25, 80)
    linkmap[9][10] = (26, 80)
    linkmap[10][9] = (31, 80)
    linkmap[9][12] = (27, 80)
    linkmap[12][9] = (35, 80)
    linkmap[9][13] = (28, 80)
    linkmap[13][9] = (38, 80)
    linkmap[11][12] = (33, 80)
    linkmap[12][11] = (36, 80)
    linkmap[11][13] = (34, 80)
    linkmap[13][11] = (39, 80)
    linkmap[12][14] = (37, 80)
    linkmap[14][12] = (42, 80)
    linkmap[13][14] = (40, 80)
    linkmap[14][13] = (43, 80)

    # traffic distrition, when non-uniform traffic is considered
    trafic_dis = [[0, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 1],
                  [2, 0, 2, 1, 8, 2, 1, 5, 3, 5, 1, 5, 1, 4],
                  [1, 2, 0, 2, 3, 2, 11, 20, 5, 2, 1, 1, 1, 2],
                  [1, 1, 2, 0, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2],
                  [1, 8, 3, 1, 0, 3, 3, 7, 3, 3, 1, 5, 2, 5],
                  [4, 2, 2, 1, 3, 0, 2, 1, 2, 2, 1, 1, 1, 2],
                  [1, 1, 11, 2, 3, 2, 0, 9, 4, 20, 1, 8, 1, 4],
                  [1, 5, 20, 1, 7, 1, 9, 0, 27, 7, 2, 3, 2, 4],
                  [2, 3, 5, 2, 3, 2, 4, 27, 0, 75, 2, 9, 3, 1],
                  [1, 5, 2, 2, 3, 2, 20, 7, 75, 0, 1, 1, 2, 1],
                  [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 2, 1, 61],
                  [1, 5, 1, 2, 5, 1, 8, 3, 9, 1, 2, 0, 1, 81],
                  [1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 0, 2],
                  [1, 4, 2, 2, 5, 2, 4, 4, 0, 1, 61, 81, 2, 0]]
    Src_Dest_Pair = []
    prob_arr = []
    computing_slice_delay_sensitive = None
    ip_slice_delay_sensitive = None
    optical_slice_delay_sensitive = None
    prob = np.array(trafic_dis) / np.sum(trafic_dis)  # 节点对之间发生业务的概率

    # generate source and destination pairs
    # for each src-dst pair, we calculate its cumlative propability based on the traffic distribution

    for ii in range(NODE_NUM):
        for jj in range(NODE_NUM):
            if ii != jj:
                prob_arr.append(prob[ii][jj])
                temp = []
                temp.append(ii + 1)
                temp.append(jj + 1)
                Src_Dest_Pair.append(temp)  # 节点对
    num_src_dest_pair = len(Src_Dest_Pair)
    prob_arr[-1] += 1 - sum(prob_arr)



    def __init__(self):
        # # 网络相关参数
        x = []
        for i in range(int(2000)):
            x.append(i)
        y = np.array(self.calcualte_ratio_nanhua(x))
        # 用3次多项式拟合
        f1 = np.polyfit(x, y, 18)
        p1 = np.poly1d(f1)
        self.yvals = np.polyval(f1, x)  # 拟合的华南业务函数

        # DRL相关
        self.alpha = 0 # 奖励中时延的参数
        # 对预测结果的修正范围
        actionRange = 0.3
        self.optical_fix_range = 4
        self.IP_fix_range = 4 # 实现的时候需要 * network.Ropt_ip = 5
        self.computing_fix_range = 2

        self.increase_delay_list = []


        self.optical_action = []
        self.IP_action = []


        self.computing_action = []
        for i in range(int((-self.optical_fix_range + 1)/2), int(self.optical_fix_range/2)):
            self.optical_action.append(i)
        n_opt = 1
        for i in range(n_opt):
            self.optical_action.append(int(999)) # 999表示不重构
        for i in range(int((-self.IP_fix_range + 1)/2), int(self.IP_fix_range/2)):
            self.IP_action.append(i)
        n_ip = 1
        for i in range(n_ip):
            self.IP_action.append(int(999))
        for i in range(int((-self.computing_fix_range + 1)/2), int(self.computing_fix_range/2)):
            self.computing_action.append(i)
        n_comput = 1
        for i in range(n_comput):
            self.computing_action.append(int(999))

        self.actions = []
        for o in self.optical_action:
            for i in self.IP_action:
                for c in self.computing_action:
                    if i == 999:
                        action = [o, 999, c]
                    else:
                        action = [o, i * self.Ropt_ip, c]
                    self.actions.append(action)

        high = np.array(
            [
                self.OPTICAL_SLOT_TOTAL,
                self.IP_SLOT_TOTAL,
                self.VM_TOTAL,
                self.OPTICAL_SLOT_TOTAL,
                self.IP_SLOT_TOTAL,
                self.VM_TOTAL,
                # float("inf"),
                # 1,
                # 1,
                # 1,
                # 1
            ],
            dtype=np.float32,
        )


        self.observation_space = spaces.Box(0, high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))

        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action):

        action = self.actions[action]
        # old_slice_optical, old_slice_ip, old_slice_comput,\
        # before_old_slice_optical, before_old_slice_ip, before_old_slice_comput, \
        # old_block, old_block_sensitive, old_block_nonsensitive, old_dealy, ExponentialSmoothing_result = self.state
        old_block =  self.num_blocks / self.req_id
        opt_ori, ip_ori, comput_ori,\
        opt_ES, ip_ES, comput_ES, \
        = self.state
        # 获得指数平滑预测得切片位置
        # optical_slice_by_ExponentialSmoothing, ip_slice_by_ExponentialSmoothing, comput_slice_by_ExponentialSmoothing = self.set_slice(
        #     [[0, self.slice_reconfig_fix(self.OPTICAL_SLOT_TOTAL, ExponentialSmoothing_result)]], [
        #         [self.slice_reconfig_fix(self.OPTICAL_SLOT_TOTAL, ExponentialSmoothing_result), self.OPTICAL_SLOT_TOTAL]],
        #     [[0, self.slice_reconfig_fix(self.IP_SLOT_TOTAL, ExponentialSmoothing_result)]],
        #     [[self.slice_reconfig_fix(self.IP_SLOT_TOTAL, ExponentialSmoothing_result), self.IP_SLOT_TOTAL]],
        #     [[0, self.slice_reconfig_fix(self.VM_TOTAL, ExponentialSmoothing_result)]],
        #     [[self.slice_reconfig_fix(self.VM_TOTAL, ExponentialSmoothing_result), self.VM_TOTAL]])
        # 指数平滑预测的切片结果

        # slice_ES = [int(self.slice_reconfig_fix(self.OPTICAL_SLOT_TOTAL, ExponentialSmoothing_result)),
        #                               int(self.slice_reconfig_fix(self.IP_SLOT_TOTAL, ExponentialSmoothing_result)),
        #                               int(self.slice_reconfig_fix(self.VM_TOTAL, ExponentialSmoothing_result))]
        slice_ES = [opt_ES, ip_ES, comput_ES]
        # 防止切片超过边界


        if action[0] == 999:
            optcial_slice_after_aciton = opt_ori
        else:
            optcial_slice_after_aciton = slice_ES[0] + action[0]
        if optcial_slice_after_aciton < 1:
            optcial_slice_after_aciton = 1
        if optcial_slice_after_aciton > self.OPTICAL_SLOT_TOTAL - 1:
            optcial_slice_after_aciton = self.OPTICAL_SLOT_TOTAL - 1

        if action[1] == 999:
            ip_slice_after_aciton = ip_ori
        else:
            ip_slice_after_aciton = slice_ES[1] + action[1]
        if ip_slice_after_aciton < 1:
            ip_slice_after_aciton = 1
        if ip_slice_after_aciton > self.IP_SLOT_TOTAL - 1:
            ip_slice_after_aciton = self.IP_SLOT_TOTAL - 1

        if action[2] == 999:
            computing_slice_after_aciton = comput_ori
        else:
            computing_slice_after_aciton = slice_ES[2] + action[2]
        if computing_slice_after_aciton < 1:
            computing_slice_after_aciton = 1
        if computing_slice_after_aciton > self.VM_TOTAL - 1:
            computing_slice_after_aciton = self.VM_TOTAL - 1

        slice_DRL = [optcial_slice_after_aciton, ip_slice_after_aciton, computing_slice_after_aciton ] # 下一状态切片结果
        # 判断终止状态
        if self.req_id >= self.NUM_REQ_MEASURE:

            output_block = self.num_blocks / self.num_req_measure
            output_opt_util = np.mean(self.resource_util_opt)
            output_ip_util = np.mean(self.resource_util_ip)
            output_memory_util = np.mean(self.resource_util_memory)
            output_delay = np.mean(self.delay_success_sum/(self.num_req_measure - self.num_blocks)) # 成功业务时延
            output_delay_due2_reconfig = self.delay_due2_slice_reconfig / (self.num_req_measure - self.num_blocks) # 成功业务重构时延
            output_delay_forward = self.delay_forward_sum / (self.num_req_measure - self.num_blocks) # 成功业务发送时延
            output_delay_propagation = self.delay_propagation_sum / (self.num_req_measure - self.num_blocks)
            output_delay_process = (self.delay_process_sum / (self.num_req_measure - self.num_blocks))
            output_block_due2_overtime = self.block_over_time / self.num_req_measure
            output_block_due2_reconfig = self.block_due2_reconfig / self.num_req_measure
            output_block_due2_memory = self.block_computing_not_enough / self.num_req_measure
            output_block_due2_opt_only = (self.block_first_path_optical_not_enough + self.block_second_path_optical_not_enough) / self.num_req_measure
            output_block_due2_IP_only = (self.block_first_path_IP_not_enough + self.block_second_path_IP_not_enough) / self.num_req_measure
            output_block_due2_optIP =(self.block_first_path_all_not_enough + self.block_second_path_all_not_enough) / self.num_req_measure
            return self.state, 0, True, self.req_block_num/self.NUM_REQ_MEASURE,\
                   output_block, output_block_due2_overtime, output_block_due2_reconfig, output_block_due2_memory, output_block_due2_opt_only, output_block_due2_IP_only, output_block_due2_optIP,\
                   output_delay, output_delay_due2_reconfig, output_delay_forward, output_delay_propagation, output_delay_process,\
                   output_opt_util, output_ip_util, output_memory_util, \
                   self.optical_reconfig_times, self.ip_reconfig_times, self.comput_reconfig_times


        # netwrok得到下一网络状态，读取状态
        next_state, optical_slice_reconfig, ip_slice_reconfig, comput_slice_reconfig, increase_time_reconfig, \
        sum_block_req_working_time_multip_data = self.get_state(slice_DRL, slice_ES, False)
        # 记录重构次数
        if optical_slice_reconfig == True:
            self.optical_reconfig_times += 1
        if ip_slice_reconfig == True:
            self.ip_reconfig_times += 1
        if comput_slice_reconfig == True:
            self.comput_reconfig_times += 1

        self.state = next_state
        #_, _, _, _, _, _,new_block, _, _, new_delay, _ = self.state
        _, _, _, _, _, _ = self.state
        new_block =  self.num_blocks / self.req_id
        # 计算奖励
        if old_block - new_block <= 0:
            r = -2 - sum_block_req_working_time_multip_data * 1e-06
            # r = - max(
            #     optical_slice_reconfig * self.delay_increase_per_optical_slice_reconfig * 1,
            #     ip_slice_reconfig * self.delay_increase_per_optical_slice_reconfig * 5,
            #     comput_slice_reconfig * self.delay_increase_per_optical_slice_reconfig * 3
            # )
            '''
            r = math.log(
                max(
                    self.delay_increase_per_optical_slice_reconfig,
                    self.delay_increase_per_ip_slice_reconfig,
                    self.delay_increase_per_comput_slice_reconfig
                ) - np.mean(self.increase_delay_list)
                # np.mean(self.increase_delay_list) /

            )
            '''

            '''
            r = - increase_time_reconfig
            '''
        else:
            # r = 1
            # r = 1 - new_block
            a = 0
            b, c, d = 1, 7, 3
            r = 1 - a * max(
                optical_slice_reconfig * self.delay_increase_per_optical_slice_reconfig * b,
                ip_slice_reconfig * self.delay_increase_per_optical_slice_reconfig * c ,
                comput_slice_reconfig * self.delay_increase_per_optical_slice_reconfig * d
            )
            # r = max(
            #     self.delay_increase_per_optical_slice_reconfig,
            #     self.delay_increase_per_ip_slice_reconfig,
            #     self.delay_increase_per_comput_slice_reconfig
            # ) - \
            #     (optical_slice_reconfig * delay_increase_per_optical_slice_reconfig +
            #      ip_slice_reconfig * delay_increase_per_optical_slice_reconfig * 5  +
            #      comput_slice_reconfig * delay_increase_per_optical_slice_reconfig * 1)

        # print(r,np.mean(self.increase_delay_list))


        # r =  - 10 * float(old_block - new_block) - 30 * (old_block_due2_opt_only - new_block_due2_opt_only)
        # r = 10 * (1 - new_block)
        # r = old_block - new_block
        # print("reward:", r, " old_block:", old_block, " mew_block:",new_block, "old_dealy:", old_dealy," new_delay:", new_delay)
        return numpy.array(next_state), r, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def get_state(self, slice_DRL, slice_ES, is_start_state): # 这里的slice是新动作的切片位置
        block_sensitive_in_epsilon = 0 # 记录本次步长中敏感业务阻塞
        block_nonsensitive_in_epsilon = 0
        req_sensitive_in_epsilon = 0 # 记录本次步长中敏感业务
        req_nonsensitive_in_epsilon = 0 # 记录本次步长中非敏感业务
        block_in_epsilon = 0  #记录本次步长中业务阻塞
        req_num_in_epsilon = 0 #记录本0次步长中业务数

        # 记录本次状态
        slice_ori = [
            self.optical_slice_delay_sensitive[0][1],
            self.ip_slice_delay_sensitive[0][1],
            self.computing_slice_delay_sensitive[0][1]
        ]

        # 根据动作切片重构
        self.set_slice(
            [[0, int(slice_DRL[0])]],
            [[int(slice_DRL[0]), self.OPTICAL_SLOT_TOTAL]],
            [[0, int(slice_DRL[1])]],
            [[int(slice_DRL[1]), self.IP_SLOT_TOTAL]],
            [[0, int(slice_DRL[2])]],
            [[int(slice_DRL[2]), self.VM_TOTAL]]
        )
        # 根据ES结果和DRL结果判断是否重构
        optical_slice_reconfig, ip_slice_reconfig, comput_slice_reconfig = self.judge_reconfig(slice_DRL, slice_ES)
        optical_slice_reconfig_before_ES, ip_slice_reconfig_before_ES, comput_slice_reconfig_before_ES = self.judge_reconfig(slice_DRL, slice_ori)

        increase_time_reconfig, sum_block_req_working_time_multip_data = \
            self.increase_delay_due2_slice_reconfig(optical_slice_reconfig_before_ES, ip_slice_reconfig_before_ES, comput_slice_reconfig_before_ES)
        # increase_time_reconfig = self.increase_delay_due2_slice_reconfig(optical_slice_reconfig, ip_slice_reconfig, comput_slice_reconfig)
        self.increase_delay_list.append(increase_time_reconfig)

        reconfiguration_flag = 0
        record_times_in_epsilon = 0 # 本次步长中记录次数
        # 为达到重构的记录次数，进行循环
        while (record_times_in_epsilon != self.reconfig_frequence):
            n = np.random.randint(10)
            self.request_initialization()
            self.req_id += 1

            self.requset_data[self.req_id] = self.current_data
            req_num_in_epsilon += 1
            self.time += self.time_to  # 记录时间

            # 根据时延动态变化业务比例
            self.RATIO_REQ = self.calcualte_ratio()
            # ratio_list.append(self.RATIO_REQ) # 用于直接预测比例

            # if self.req_id % 1000 == 0:
            #     print(self.req_id, self.RATIO_REQ, self.time, self.time_to)

            if n < self.RATIO_REQ * 10:
                self.req_type = "sensitive"
                # self.max_delay = self.delay_idea * self.OVERTIME_TOLERANCE + self.OEO_delay_for_delay_sensitive
                self.max_delay = self.delay_idea * self.over_time_sensitive

                req_sensitive_in_epsilon += 1
            else:
                self.req_type = "nonsensitive"
                # self.max_delay = self.delay_idea * self.OVERTIME_TOLERANCE * 5 + self.OEO_delay_for_delay_sensitive # 假设非时延敏感时延容忍*5
                # self.max_delay = self.delay_idea * self.OVERTIME_TOLERANCE_for_nonsensitive + self.OEO_delay_for_delay_sensitive  # 假设非时延敏感时延容忍*5
                self.max_delay = self.delay_idea
                req_nonsensitive_in_epsilon += 1

            if self.req_type == "sensitive":
                self.req_num_sensitive += 1
                self.req_sensitive.append(self.req_id)
            else:
                self.req_num_nonsensitive += 1
                self.req_nonsensitive.append(self.req_id)

            # 资源释放
            (self.slot_map, self.ip_slot_map, self.request_set_optical, self.request_set_ip,
             self.request_set_computing, self.slot_map_t,
             self.ip_slot_map_t) = self.release(
                self.slot_map, self.ip_slot_map, self.request_set_optical, self.request_set_ip,
                self.request_set_computing, self.slot_map_t,
                self.ip_slot_map_t, self.time_to)

            # paths_record = [9999999999] #记录可用路径 【时延，路径1，路径2】
            for rr in range(self.kpath):  # kpath=5
                for dd in range(self.kpath):
                    self.path_id_first = rr // self.M  # path to use M=1
                    self.path_id_second = dd // self.M  # path to use M=1
                    self.FS_id_first = math.fmod(rr,
                                                 self.M)  # the FS_id's available FS-block to use 求rr除以M后的余数，结果为浮点型 FS_id = 0 表示first fit
                    self.FS_id_second = math.fmod(dd, self.M)  # 第二条路

                    self.slice_piece = np.random.randint(0, len(self.optical_slice_delay_sensitive))  # 随机选择切片块，之后改
                    self.slice_piece = 0  # 选择第0个切片块
                    if self.req_type == "sensitive":
                        # optical_slice_delay_sensitive ip_slice_delay_sensitive是不是可以删去
                        # 输出的DC是从1开始的
                        self.paths_record, self.DC, self.path_first, self.path_second, self.num_FS, self.delay_forward, self.delay_propagation, self.delay_process, self.current_TTL, self.data_for_each_VM, self.block_type = self.selectDC_calcualteTime_sensitive(
                            self.current_src, self.current_dst, self.slice_piece, self.computing_queue,
                            self.Candidate_Paths, self.path_id_first, self.path_id_second, self.current_bandwidth,
                            self.current_data, self.delay_idea, self.ip_slot_map, self.ip_slot_map_t, self.slot_map,
                            self.slot_map_t, self.FS_id_first, self.optical_slice_delay_sensitive,
                            self.ip_slice_delay_sensitive, self.computing_slice_delay_sensitive, self.FS_id_second,
                            self.DC_strategy)

                    elif self.req_type == "nonsensitive":
                        self.paths_record, self.DC, self.path_first, self.path_second, self.num_FS, self.delay_forward, self.delay_propagation, self.delay_process, self.current_TTL, self.data_for_each_VM, self.block_type = self.selectDC_calcualteTime_nonsensitive(
                            self.current_src, self.current_dst, self.slice_piece, self.computing_queue,
                            self.Candidate_Paths, self.path_id_first, self.path_id_second, self.current_bandwidth,
                            self.current_data, self.delay_idea, self.ip_slot_map, self.ip_slot_map_t, self.slot_map,
                            self.slot_map_t, self.FS_id_first, self.optical_slice_delay_nonsensitive,
                            self.ip_slice_delay_nonsensitive, self.computing_slice_delay_nonsensitive,
                            self.FS_id_second, self.DC_strategy)
                    # if len(self.paths_record) == 1:
                    #     continue
                    if len(self.paths_record) != 1:
                        break
                if len(self.paths_record) != 1:
                    break
            if len(self.paths_record) != 1:
                self.current_TTL = self.paths_record[0]
                self.path_links_first = self.paths_record[1]
                self.fs_start_first = self.paths_record[2]
                self.fs_end_first = self.paths_record[3]
                self.path_links_second = self.paths_record[4]
                self.fs_start_second = self.paths_record[5]
                self.fs_end_second = self.paths_record[6]
                self.num_FS_ip = self.paths_record[7]
                self.current_src = self.paths_record[8]
                self.DC = self.paths_record[9]
                self.current_dst = self.paths_record[10]
                self.delay_process = self.paths_record[11]

                # 根据最短结果占用光资源
                # 这时候slot_map_t有1了

                self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(self.slot_map,
                                                                                        self.paths_record[1],
                                                                                        self.paths_record[2],
                                                                                        self.paths_record[3],
                                                                                        self.slot_map_t,
                                                                                        self.paths_record[
                                                                                            0])  # 在网络中占用第一条路径光资源

                self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(self.slot_map,
                                                                                        self.paths_record[4],
                                                                                        self.paths_record[5],
                                                                                        self.paths_record[6],
                                                                                        self.slot_map_t,
                                                                                        self.paths_record[
                                                                                            0])  # 在网络中占用第二条路径光资源
                # 节点从1开始，需要-1
                # 更新IP资源，返回该请求IP隙占用情况，0为占用，1为可用
                ip_slot_map, ip_slot_map_t, ip_FS_used_in_the_request_first = self.update_ip_slot_map_for_committing_wp(
                    self.paths_record[8] - 1, self.paths_record[9] - 1, self.paths_record[7], self.ip_slot_map,
                    self.ip_slot_map_t, self.paths_record[0])  # 在网络中占用第一条路径IP资源
                ip_slot_map, ip_slot_map_t, ip_FS_used_in_the_request_second = self.update_ip_slot_map_for_committing_wp(
                    self.paths_record[9] - 1, self.paths_record[10] - 1, self.paths_record[7], self.ip_slot_map,
                    self.ip_slot_map_t, self.paths_record[0])  # 在网络中占用第二条路径IP资源

                # 占用计算存储资源
                computing_queue = self.update_computing_for_committing(self.computing_queue, self.DC, self.data_for_each_VM)

                temp_ = []  # 记录业务信息
                path_links = []
                path_links.append(self.paths_record[1])  # 多段path [path1, path2]
                path_links.append(self.paths_record[4])
                # print("path_links", path_links)
                temp_.append(list(path_links))

                fs_start_end = []
                first_path = []
                first_path.append(self.paths_record[2])  # 多个频隙段[ [s,d], [s,d] ]
                first_path.append(self.paths_record[3])
                fs_start_end.append(first_path)
                second_path = []
                second_path.append(self.paths_record[5])
                second_path.append(self.paths_record[6])
                fs_start_end.append(second_path)
                temp_.append(list(fs_start_end))
                # temp_ = [ [第一条路， 第二条路]，[ [ 第一条路start fs, 第一条路end fs], [第二条路start fs, 第二条路end fs] ]  ]
                temp_.append(self.paths_record[0])
                # print("temp_", temp_)
                self.request_set_optical[self.req_id] = temp_

                ip_FS_used_in_the_request = []
                ip_FS_used_in_the_request.append(ip_FS_used_in_the_request_first)  # 多段IP隙 [ listIP, listIP ]
                ip_FS_used_in_the_request.append(ip_FS_used_in_the_request_second)
                # 【第一段路径占用的IP隙（ip_slot_map的形式），第二段路径占用的IP隙（ip_slot_map的形式）】
                self.request_set_ip[self.req_id] = ip_FS_used_in_the_request

                used_VM = []
                for i in range(len(self.data_for_each_VM)):
                    if self.data_for_each_VM[i] != 0:
                        used_VM.append(1)  # 1表示被该业务占用
                    else:
                        used_VM.append(0)  # 0表示没有被该业务占用

                # 【DC, 使用的VM, 每个vm的数据量，原本业务持续时间, 第一段路径时间】（1记为使用）
                self.request_set_computing[self.req_id] = [self.DC, used_VM, self.data_for_each_VM, self.paths_record[0],
                                                           self.first_path_delay_propagation]

                self.request_set_delay[self.req_id] = [self.max_delay, self.paths_record[0]]

                delay = self.paths_record[0]

            else:  # 所有K条路都不行
                delay = self.TMAX
                block = 1
                self.req_block_num += 1
                block_in_epsilon += 1
            # if self.req_id > 0.3 * self.num_req_measure:

            # 记录发生阻塞，延迟，资源利用
            if self.req_type == "sensitive":
                # self.sensitive_load += self.delay_idea * self.current_data # 记录负载 不使用
                self.sensitive_times += 1  # 记录次数
            elif self.req_type == "nonsensitive":
                # self.nonsensitive_load += self.delay_idea * self.current_data
                self.nonsensitive_times += 1

            self.record_time += self.time_to

            # 记录业务发生频率
            if self.record_time - self.time_record_req_load >= 0:
                record_times_in_epsilon += 1
                # self.record_load()
                self.record_req_times()

            self.delay_sum += delay
            # self.delay_idea_sum += self.delay_idea + self.OEO_delay_for_delay_sensitive
            if delay != self.TMAX:  # 传输成功
                self.num_success += 1
                self.delay_success_sum += delay
                self.delay_propagation_sum += self.delay_propagation
                self.delay_forward_sum += self.delay_forward
                self.delay_process_sum += self.delay_process
                self.delay_idea_sum += self.delay_idea
            else:  # 不成功
                self.num_blocks += block
                if self.req_type == "sensitive":
                    self.sensitive_block += block
                    block_sensitive_in_epsilon += 1
                else:
                    self.nonsensitive_block += block
                    block_nonsensitive_in_epsilon += 1

                self.delay_block_sum += delay
                if self.block_type == "first path block, optical not enough":
                    self.block_first_path_optical_not_enough += 1
                elif self.block_type == "first path block, IP not enough":
                    self.block_first_path_IP_not_enough += 1
                elif self.block_type == "first path block, all not enough":
                    self.block_first_path_all_not_enough += 1
                elif self.block_type == "second path block, optical not enough":
                    self.block_second_path_optical_not_enough += 1
                elif self.block_type == "second path block, IP not enough":
                    self.block_second_path_IP_not_enough += 1
                elif self.block_type == "second path block, all not enough":
                    self.block_second_path_all_not_enough += 1
                elif self.block_type == "over time":
                    self.block_over_time += 1
                elif self.block_type == "computing not enough":
                    self.block_computing_not_enough += 1

                self.resource_util_opt.append(
                    1 - np.sum(self.slot_map) / (self.LINK_NUM * self.OPTICAL_SLOT_TOTAL))  # 1代表可用，
                self.resource_util_ip.append(1 - np.sum(self.ip_slot_map) / (self.NODE_NUM * self.IP_SLOT_TOTAL))
                self.resource_util_memory.append(np.sum(self.computing_queue) / (
                        self.NODE_NUM * self.VM_MEMORY * self.VM_TOTAL))

        reconfiguration_flag = self.num_record_req_load

        ratio_pre_dou, ratio_pre_tri, ratio_list = self.prediction_ratio_DRL(self.num_pre_record)
        ratio = ratio_pre_tri

        opt_ori = slice_DRL[0]
        ip_ori = slice_DRL[1]
        comput_ori = slice_DRL[2]

        state_block = self.num_blocks / self.req_id
        state_sensitive_block = self.sensitive_block / self.req_num_sensitive
        state_nonsensitive_block = self.nonsensitive_block / self.req_num_nonsensitive

        state_delay = self.delay_success_sum / self.delay_idea_sum # 归一化的超时delay 成功业务时延/成功业务理想时延


        state_ExponentialSmoothing_result = ratio

        opt_ES, ip_ES, comput_ES = int(self.slice_reconfig_fix(self.OPTICAL_SLOT_TOTAL, state_ExponentialSmoothing_result)), \
                                   int(self.slice_reconfig_fix(self.IP_SLOT_TOTAL, state_ExponentialSmoothing_result)),\
                                   int(self.slice_reconfig_fix(self.VM_TOTAL, state_ExponentialSmoothing_result))

        # next_state = (opt_ori, ip_ori, comput_ori,
        #               opt_ES, ip_ES, comput_ES,
        #               state_block, state_sensitive_block, state_nonsensitive_block, state_delay, state_ExponentialSmoothing_result)
        opt_reconfig = int(optical_slice_reconfig_before_ES)
        ip_reconfig = int(ip_slice_reconfig_before_ES)
        comput_reconfig = int(comput_slice_reconfig_before_ES)
        block_due2_opt_only = (self.block_first_path_optical_not_enough + self.block_second_path_optical_not_enough) / self.num_req_measure
        next_state = (
            opt_ori, ip_ori, comput_ori,
            opt_ES, ip_ES, comput_ES,
            # increase_time_reconfig,
            # state_block
        )
        return next_state, optical_slice_reconfig_before_ES, ip_slice_reconfig_before_ES, \
               comput_slice_reconfig_before_ES, increase_time_reconfig, sum_block_req_working_time_multip_data



    def reset(self):
        # 两个文件的内容记录在内存里
        self.sensitive_req_load = []
        self.nonsensitive_req_load = []

        self.optical_reconfig_times = 0 # 切片重构次数
        self.ip_reconfig_times = 0
        self.comput_reconfig_times = 0
        # self.
        # = 1

        self.Candidate_Paths = self.calcualte_Candidate_Paths(self.linkmap, self.NODE_NUM, self.kpath)  # 计算kpath条路径

        # initiate the EON
        self.network_initialization()  # 在最开始初始化网络
        self.set_slice([[0, self.slice_reconfig_fix(self.OPTICAL_SLOT_TOTAL, self.INIT_RATIO)]], [
            [self.slice_reconfig_fix(self.OPTICAL_SLOT_TOTAL, self.INIT_RATIO), self.OPTICAL_SLOT_TOTAL]],
                       [[0, self.slice_reconfig_fix(self.IP_SLOT_TOTAL, self.INIT_RATIO)]], [
                           [self.slice_reconfig_fix(self.IP_SLOT_TOTAL, self.INIT_RATIO),
                            math.ceil(self.IP_SLOT_TOTAL)]],
                       [[0, self.slice_reconfig_fix(self.VM_TOTAL, self.INIT_RATIO)]],
                       [[self.slice_reconfig_fix(self.VM_TOTAL, self.INIT_RATIO), self.VM_TOTAL]])

        # 切片重构相关参数
        self.time = 0  # 记录时间
        self.sensitive_load = 0  # 一段时间的负载
        self.nonsensitive_load = 0  # 一段时间的负载
        self.sensitive_times = 0  # 一段时间的业务次数
        self.nonsensitive_times = 0  # 一段时间的业务次数

        self.num_record_req_load = 0  # 文件中记录业务负载的数量

        self.num_pre_record = 1  # 预测num_pre_record次负载，去平均
        reconfiguration_flag = 0  # 重构标记，避免多次重构
        self.record_time = 0
        self.req_block_num = 0


        # 先生成一部分数据
        i = 0
        while i < 50:

            n = np.random.randint(10)
            self.request_initialization()
            self.time += self.time_to
            # 根据时延动态变化业务比例
            self.RATIO_REQ = self.calcualte_ratio()
            if n < self.RATIO_REQ * 10:
                self.req_type = "sensitive"
            else:
                self.req_type = "nonsensitive"
            if self.req_type == "sensitive":
                self.sensitive_times += 1
            elif self.req_type == "nonsensitive":
                self.nonsensitive_times += 1
            self.record_time += self.time_to
            if self.record_time - self.time_record_req_load >= 0:
                self.record_req_times()
                i += 1



        self.state, optical_slice_reconfig,ip_slice_reconfig,comput_slice_reconfig, increase_time_reconfig, sum_block_req_working_time_multip_data = self.get_state(
            [self.OPTICAL_SLOT_TOTAL/2, self.IP_SLOT_TOTAL/2, self.VM_TOTAL/2],
            [self.OPTICAL_SLOT_TOTAL / 2, self.IP_SLOT_TOTAL / 2, self.VM_TOTAL / 2],

            True)

        return numpy.array(self.state)






    def calcualte_Candidate_Paths(self, linkmap, nodeNum, k):  # 计算K条路径
        G = nx.DiGraph()
        Candidate_Paths = defaultdict(
            lambda: defaultdict(lambda: defaultdict(
                lambda: None)))  # Candidate_Paths[i][j][k]:the k-th path from i to j
        for i in range(nodeNum):
            for j in range(nodeNum):
                if i != j:
                    if linkmap[i + 1][j + 1] != None:
                        G.add_edge(i + 1, j + 1, length=1, weight=linkmap[i + 1][j + 1][1])
        for i in range(nodeNum):
            for j in range(nodeNum):
                if i != j:
                    paths = k_shortest_paths(G, i + 1, j + 1, k, "weight")
                    for p in range(len(paths[0])):
                        Candidate_Paths[i + 1][j + 1][p] = paths[0][p]
        return Candidate_Paths

    def _get_path(self, src, dst, Candidate_Paths, k):  # get path k of from src->dst
        if src == dst:
            print('error: _get_path()')
            path = []
        else:
            path = Candidate_Paths[src][dst][k]
            if path is None:
                return None
        return path

    def calclink(self, p):  # map path to links 路径中所有link的编号
        path_link = []
        for a, b in zip(p[:-1],
                        p[1:]):  # line = "abcde" line[:-1] 结果为：'abcd'      print(a[1:]) —>  [2, 3, 4, 5] zip:两两组合
            # a b 是节点对
            k = self.linkmap[a][b][0]
            path_link.append(k)
        return path_link

    def get_new_slot_temp(self, slot_temp, path_link,
                          slot_map):  # 返回被占用的slot 因为遵循频谱一致性，需要找到path_link中所有link均空闲的slot，记为0
        for i in path_link:
            for j in range(self.OPTICAL_SLOT_TOTAL):
                slot_temp[j] = slot_map[i][j] & slot_temp[j]  # 1&1 = 1
        return slot_temp

    # only used when we apply heuristic algorithms
    def mark_vector(self, vector, default, ii, le):
        # return 几段可用频率，起点，长度。 [1,0,1,1] —> (2, [0, 2], [1, 2])
        flag = 0
        slotscontinue = []
        slotflag = []
        while ii <= le - 1:
            tempvector = vector[ii:le]  # 从ii开始截取slot_temp
            default_counts = tempvector.count(default)  # 可用频隙数量
            if default_counts == 0:  # 没有可用频隙
                break
            else:
                a = tempvector.index(default)  # 返回首次出现的位置
                ii += a  # 跳过前面的0
                flag += 1
                slotflag.append(ii)
                m = vector[ii + 1:le]
                m_counts = m.count(1 - default)  # 计算之后0的数量
                if m_counts != 0:
                    n = m.index(1 - default)
                    slotcontinue = n + 1
                    slotscontinue.append(slotcontinue)
                    ii += slotcontinue
                else:  # 之后都是1
                    slotscontinue.append(le - ii)  # 都加入slotscontinue
                    break
        return flag, slotflag, slotscontinue

    # ---------------------------------------------------------------------------------

    # IP资源相关

    def judge_ip_availability(self, current_src, current_dst, num_FS_ip, ip_slot_map, ip_slice_range_group):
        """
        判断IP隙是否足够,所有切片块资源之和够即可

        :param current_src:
        :param current_dst:
        :param num_FS_ip: IP隙需求
        :param ip_slot_map:
        :param ip_slice_range_group: 整个一类IP切片的集合，不连续的IP切片可以绑定使用。ABA两类交叉排布，两个A可以一起使用
        :return: IP资源是否足够
        """
        ip_flag = 0
        src_FS = 0
        dst_FS = 0
        for slice_range in ip_slice_range_group:
            src_FS += ip_slot_map[current_src][slice_range[0]: slice_range[1]].count(1)
            dst_FS += ip_slot_map[current_dst][slice_range[0]: slice_range[1]].count(1)
        if (src_FS >= num_FS_ip and dst_FS >= num_FS_ip):
            ip_flag = 1
            return ip_flag
        # if ip_flag == 0:
        #     print(self.req_type,ip_slice_range_group, len(ip_slot_map[current_src]))
        return ip_flag

    def update_ip_slot_map_for_committing_wp(self, current_src, current_dst, num_FS_ip, ip_slot_map, ip_slot_map_t,
                                             current_TTL):  # 更新ip隙，记录占用的IP隙为0
        """

        :param current_src:
        :param current_dst:
        :param num_FS_ip: =0
        :param ip_slot_map:
        :param ip_slot_map_t:
        :param current_TTL:
        :return: 更新后的maps
        """
        f = num_FS_ip
        ip_FS_used_in_the_request = [[1 for x in range(self.IP_SLOT_TOTAL)] for y in range(self.NODE_NUM)] # 记录占用情况，0被占用
        if self.req_type == "sensitive":
            slice = self.ip_slice_delay_sensitive[self.slice_piece]
        else:
            slice = self.ip_slice_delay_nonsensitive[self.slice_piece]

        while f > 0:

            for i in range(slice[1] - slice[0]):
                if ip_slot_map[current_src][slice[0] + i] == 1:
                    ip_slot_map[current_src][slice[0] + i] = 0
                    ip_slot_map_t[current_src][slice[0] + i] = current_TTL
                    ip_FS_used_in_the_request[current_src][slice[0] + i] = 0
                    f = f - 1
                    if f == 0:
                        break
        f = num_FS_ip
        while f > 0:
            for j in range(slice[1] - slice[0]):
                if ip_slot_map[current_dst][slice[0] + j] == 1:
                    ip_slot_map[current_dst][slice[0] + j] = 0
                    ip_slot_map_t[current_dst][slice[0] + j] = current_TTL
                    ip_FS_used_in_the_request[current_dst][slice[0] + j] = 0
                    f = f - 1
                    if f == 0:
                        break
        return ip_slot_map, ip_slot_map_t, ip_FS_used_in_the_request

    def update_ip_slot_map_for_releasing_wp(self, ip_slot_map,
                                            requset_set_ip_rr):  # update ip_slotmap, mark released ip_slot as free
        """

        :param ip_slot_map:
        :param requset_set_ip_rr: requset_set_ip_rr[n][s]第N个光路的第S个link，进行资源释放
        :return: 释放后的ip_slot_map
        """
        for n in range(self.NODE_NUM):
            for s in range(self.IP_SLOT_TOTAL):
                if requset_set_ip_rr[n][s] == 0:
                    if ip_slot_map[n][s] != 0:
                        print('Error--update_ip_slot_map_for_releasing_wp!------------------------------------')
                    else:
                        ip_slot_map[n][s] = 1
        return ip_slot_map

    # ---------------------------------------------------------------------------------

    # 光资源相关

    def judge_optical_availability(self, slot_temp, current_slots, FS_id, slice):
        """
        判断光资源是否可用，依次判断每个切片块资源是否足够
        :param slot_temp: 多个links都空闲的频谱 1可用
        :param current_slots: 需要的连续频谱数量
        :param FS_id: =0，first fit 参数，第一个可用的连续slot
        :param slice: 切片块（切片[切片块编号])
        :return: flag_availability是否可用，频谱起点fs，终点fe
        """
        fs = -1
        fe = -1
        flag_availability = 0
        flag, slotflag, slotscontinue = self.mark_vector(slot_temp, 1, slice[0], slice[1])
        if flag > 0:  # 有可用频谱
            nn = len(slotscontinue)
            t = 0
            for i in range(nn):
                if slotscontinue[i] >= current_slots:  # 频谱长度足够传输业务
                    if t == FS_id:  # FS_id = 0 for first fit 选择第一个可用slot block
                        fs = slotflag[i]
                        fe = slotflag[i] + current_slots - 1
                        flag_availability = 1
                        return flag_availability, fs, fe
                    t += 1
            return flag_availability, fs, fe
        else:  # 没有可用频谱 返回0，-1，-1
            flag_availability = 0
        return flag_availability, fs, fe

    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe, slot_map_t,
                                          current_TTL):  # update slotmap, mark allocated FS' as occupied
        """
        占用频谱资源
        :param slot_map: 光频谱map
        :param current_wp_link: req占用的links编号
        :param current_fs: 频谱起点
        :param current_fe: 频谱终点
        :param slot_map_t: 光频谱占用时间map
        :param current_TTL: req持续时间
        :return: 更新后的slot_map，slot_map_t
        """
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):

                if s >= len(slot_map_t[ll]) or s < 0:
                    print(s, len(slot_map_t[ll]))

                if slot_map[ll][s] != 1 or slot_map_t[ll][s] != 0:  # means error 已被占用
                    print("means error", "link", ll, "频隙起点", current_fs, "频隙终点",current_fe, "slot_map_t",slot_map_t[ll], s, "slot_map[ll][s]", slot_map[ll][s])
                else:  # still unused
                    slot_map[ll][s] = 0  # 占用
                    slot_map_t[ll][s] = current_TTL  # 持续时间
        return slot_map, slot_map_t

    def update_slot_map_for_releasing_wp(self, slot_map, current_wp_link, current_fs,
                                         current_fe):  # update slotmap, mark released FS' as free
        """
        释放光频谱资源 置位1 可用
        :param slot_map: 光频谱map
        :param current_wp_link: req占用的links编号
        :param current_fs: 光频谱起点
        :param current_fe: 光频谱终点
        :return: 更新后的slot_map
        """
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):
                if slot_map[ll][
                    s] != 0:  # this FS should be occupied by current request, !=0 means available now, which is wrong
                    print('Error--update_slot_map_for_releasing_wp!')
                else:  # still unused
                    slot_map[ll][s] = 1
                    # print("释放[ll][s],link slot", ll, s)
        return slot_map

    def release(self, slot_map, ip_slot_map, request_set_optical, request_set_ip, request_set_computing, slot_map_t,
                ip_slot_map_t, time_to):  # update slotmap to release FS' occupied by expired requests
        """

        :param slot_map:
        :param ip_slot_map:
        :param request_set_optical: 记录业务ID
        :param request_set_ip: 记录业务ID
        :param request_set_computing: 记录业务ID
        :param slot_map_t:
        :param ip_slot_map_t:
        :param time_to: 时间频隙时间
        :return: 更新后的map
        """
        if request_set_optical:
            # update slot_map_t 更新每个频隙持续占用时间
            for ii in range(self.LINK_NUM):
                for jj in range(self.OPTICAL_SLOT_TOTAL):
                    if slot_map_t[ii][jj] > time_to:
                        slot_map_t[ii][jj] -= time_to  # 减去time_to
                    elif slot_map_t[ii][jj] > 0:
                        slot_map_t[ii][jj] = 0
            # update ip_slot_map_t 更新每个IP隙持续占用时间
            for ii in range(self.NODE_NUM):
                for jj in range(self.IP_SLOT_TOTAL):
                    if ip_slot_map_t[ii][jj] > time_to:
                        ip_slot_map_t[ii][jj] -= time_to  # 减去time_to
                    elif ip_slot_map_t[ii][jj] > 0:
                        ip_slot_map_t[ii][jj] = 0

            # 更新计算资源
            for ii in range(self.NODE_NUM):
                for jj in range(self.VM_TOTAL):
                    if self.computing_queue[ii][jj] > time_to * self.vm_process_speed:
                        self.computing_queue[ii][
                            jj] -= time_to * self.vm_process_speed  # 减去time_to * vm_process_speed,更新数据队列
                    elif self.computing_queue[ii][jj] > 0:
                        self.computing_queue[ii][jj] = 0
            del_id = []
            for rr in request_set_optical:
                request_set_optical[rr][2] -= time_to  # 业务持续时间减去time_to
                if request_set_optical[rr][2] <= 0:  # 到时业务
                    # print("到时业务",rr, self.request_set_optical[rr])
                    for kk in range(len(request_set_optical[rr][0])):  # 对每一段路释放光资源
                        current_wp_link = request_set_optical[rr][0][kk]
                        fs_wp = request_set_optical[rr][1][kk][0]
                        fe_wp = request_set_optical[rr][1][kk][1]
                        slot_map = self.update_slot_map_for_releasing_wp(slot_map, current_wp_link, fs_wp,
                                                                         fe_wp)  # 至为1，可用
                        ip_slot_map = self.update_ip_slot_map_for_releasing_wp(ip_slot_map,
                                                                               request_set_ip[rr][kk])  # 更新IP隙
                    del_id.append(rr)
            for ii in del_id:
                del request_set_optical[ii]  # 删除所有到时间的request
                del request_set_ip[ii]
                del request_set_computing[ii]
        return slot_map, ip_slot_map, request_set_optical, request_set_ip, request_set_computing, slot_map_t, ip_slot_map_t

    # 判断资源可用
    def judge_ability(self, paths_record, optical_slice, current_TTL, delay_idea, current_bandwidth,
                      ip_slot_map, ip_slot_map_t, slot_map, slot_map_t, num_FS, path_first, FS_id_first, slice_piece,
                      current_src,
                      DC, ip_slice, path_second, FS_id_second, current_dst, delay_process, computing_flag):
        """

        :param paths_record: 应该为初始化的99999
        :param optical_slice: 时延敏感光切片
        :param current_TTL:
        :param delay_idea:
        :param current_bandwidth:
        :param ip_slot_map:
        :param ip_slot_map_t:
        :param slot_map:
        :param slot_map_t:
        :param num_FS:
        :param path_first: 路径编号
        :param FS_id_first: 首次命中 = 0
        :param slice_piece:
        :param current_src:
        :param DC: 数据中心编号
        :param ip_slice:
        :param path_second:
        :param FS_id_second:
        :param current_dst:
        :param delay_process:
        :param computing_flag: 计算资源是否足够
        :return: 返回路径信息paths_record：current_TTL, path_links_first, fs_start_first, fs_end_first, path_links_second,
                                    fs_start_second,fs_end_second, num_FS_ip, current_src, DC, current_dst,
                                    delay_process
                                    阻塞信息
        """
        if computing_flag == 0:
            block_type = "computing not enough"
            return paths_record, block_type  # 计算资源不足
        if current_TTL > (self.max_delay):
            block_type = "over time"
            # print("over time",self.req_id, self.req_type, self.delay_process, np.sum(self.computing_queue))
            return paths_record, block_type  # 超时退出， 时延非敏感业务不判断
        num_FS_ip = math.ceil(current_bandwidth / (self.BFS / self.Ropt_ip))  # 两条路占用频谱数量一样

        # 判断并临时占用资源
        ip_slot_map_first = self.copy_list(ip_slot_map)
        ip_slot_map_t_first = self.copy_list(ip_slot_map_t)
        slot_map_first = self.copy_list(slot_map)
        slot_map_t_first = self.copy_list(slot_map_t)

        slot_temp_first = [1] * self.OPTICAL_SLOT_TOTAL  # 120个1
        path_links_first = self.calclink(path_first)  # 路径中link编号
        slot_temp_first = self.get_new_slot_temp(slot_temp_first, path_links_first,
                                                 slot_map_first)  # spectrum utilization on the whole path 返回path_links均空闲的slot 满足一致性
        opt_flag_first, fs_start_first, fs_end_first = self.judge_optical_availability(
            slot_temp_first, num_FS, FS_id_first, optical_slice[slice_piece])  # flag = 1 有可用， =0 无可用

        if fs_start_first < 0 & opt_flag_first != 0:
            print("-------------------------------------------fs_start_first<0--------------------------")

        ip_flag_first = self.judge_ip_availability(current_src - 1, DC - 1, num_FS_ip, ip_slot_map_first,
                                                   ip_slice)  # 节点从1开始 IP无切片块
        # computing_flag, data_for_each_VM  = judge_computing_ability(DC - 1, current_data, path_first, computing_queue, slice_piece) # 节点从1开始
        if (ip_flag_first == 0 or opt_flag_first == 0):  # 如果第一条路失败
            if (opt_flag_first == 0 and ip_flag_first == 1):
                block_type = "first path block, optical not enough"
                return paths_record, block_type
            elif (opt_flag_first == 1 and ip_flag_first == 0):
                block_type = "first path block, IP not enough"
                return paths_record, block_type
            else:
                block_type = "first path block, all not enough"
                return paths_record, block_type

        else:  # 第一条路满足要要求，判断第二条路
            # 临时占用资源
            slot_map_first, slot_map_t_first = self.update_slot_map_for_committing_wp(
                slot_map_first, path_links_first, fs_start_first, fs_end_first, slot_map_t_first,
                1)  # 临时占用第一条路径的光资源，用于判断第二条路
            ip_slot_map_first, ip_slot_map_t_first, ip_FS_used_in_the_request_first_temp = self.update_ip_slot_map_for_committing_wp(
                current_src - 1, DC - 1, num_FS_ip, ip_slot_map_first, ip_slot_map_t_first, 1)  # 临时占用第一条路径的IP资源
            ip_slot_map_second = ip_slot_map_first
            slot_map_second = slot_map_first

            slot_temp_second = [1] * self.OPTICAL_SLOT_TOTAL  # 100个1
            path_links_second = self.calclink(path_second)  # 路径中link编号
            slot_temp_second = self.get_new_slot_temp(slot_temp_second, path_links_second,
                                                      slot_map_second)  # spectrum utilization on the whole path 返回path_links均空闲的slot 满足一致性
            opt_flag_second, fs_start_second, fs_end_second = self.judge_optical_availability(
                slot_temp_second, num_FS, FS_id_second,
                optical_slice[slice_piece])  # flag = 1 有可用， =0 无可用
            ip_flag_second = self.judge_ip_availability(DC - 1, current_dst - 1, num_FS_ip, ip_slot_map_second,
                                                        ip_slice)  # 节点从1开始
            if opt_flag_second == 1 and ip_flag_second == 1:
                if float(current_TTL) < float(paths_record[0]):
                    paths_record = [current_TTL, path_links_first, fs_start_first, fs_end_first, path_links_second,
                                    fs_start_second,
                                    fs_end_second, num_FS_ip, current_src, DC, current_dst,
                                    delay_process]  # 更新最短路记录 time shortest 循环计算
        if len(paths_record) != 1:
            block_type = "no block"
            return paths_record, block_type
        else:
            if opt_flag_second == 0 and ip_flag_second == 1:
                block_type = "second path block, optical not enough"
                return paths_record, block_type
            elif opt_flag_second == 1 and ip_flag_second == 0:
                block_type = "second path block, IP not enough"
                return paths_record, block_type
            else:
                block_type = "second path block, all not enough"
                return paths_record, block_type

    # ---------------------------------------------------------------------------------

    # 数据中心相关

    def selectDC_calcualteTime_sensitive(self, current_src, current_dst, slice_piece, computing_queue,
                                         Candidate_Paths, path_id_first, path_id_second, current_bandwidth,
                                         current_data, delay_idea, ip_slot_map, ip_slot_map_t, slot_map, slot_map_t,
                                         FS_id_first,
                                         optical_slice_delay_sensitive, ip_slice_delay_sensitive,
                                         computing_slice_delay_sensitive, FS_id_second, strategy="random"):

        """

        :param current_src:
        :param current_dst:
        :param slice_piece:
        :param computing_queue:
        :param Candidate_Paths:
        :param path_id_first:
        :param path_id_second:
        :param current_bandwidth:
        :param current_data:
        :param delay_idea:
        :param ip_slot_map:
        :param ip_slot_map_t:
        :param slot_map:
        :param slot_map_t:
        :param FS_id_first:
        :param optical_slice_delay_sensitive:
        :param ip_slice_delay_sensitive:
        :param computing_slice_delay_sensitive:
        :param FS_id_second: =0
        :param strategy:
        :return: paths_record: current_TTL, path_links_first, fs_start_first, fs_end_first, path_links_second,
                                    fs_start_second,fs_end_second, num_FS_ip, current_src, DC, current_dst,
                                    delay_process
        """
        # 选择DC，计算时间
        paths_record, DC, path_first, path_second, num_FS, delay_forward, delay_propagation, delay_process, current_TTL, data_for_each_VM, block_type = self.select_DC(
            current_src, current_dst, slice_piece, computing_queue, Candidate_Paths, path_id_first, path_id_second,
            current_bandwidth, current_data, delay_idea, ip_slot_map, ip_slot_map_t, slot_map, slot_map_t, FS_id_first,
            optical_slice_delay_sensitive, ip_slice_delay_sensitive, computing_slice_delay_sensitive, FS_id_second,
            strategy)

        return paths_record, DC, path_first, path_second, num_FS, delay_forward, delay_propagation, delay_process, current_TTL, data_for_each_VM, block_type

    def selectDC_calcualteTime_nonsensitive(self, current_src, current_dst, slice_piece, computing_queue,
                                            Candidate_Paths, path_id_first, path_id_second, current_bandwidth,
                                            current_data, delay_idea, ip_slot_map, ip_slot_map_t, slot_map, slot_map_t,
                                            FS_id_first,
                                            optical_slice_delay_nonsensitive, ip_slice_delay_nonsensitive,
                                            computing_slice_delay_nonsensitive, FS_id_second, strategy="random"):
        """
        :param current_src:
        :param current_dst:
        :param slice_piece:
        :param computing_queue:
        :param Candidate_Paths:
        :param path_id_first:
        :param path_id_second:
        :param current_bandwidth:
        :param current_data:
        :param delay_idea:
        :param ip_slot_map:
        :param ip_slot_map_t:
        :param slot_map:
        :param slot_map_t:
        :param FS_id_first:
        :param optical_slice_delay_nonsensitive:
        :param ip_slice_delay_nonsensitive:
        :param computing_slice_delay_nonsensitive:
        :param FS_id_second: =0
        :param strategy:
        :return: paths_record: current_TTL, path_links_first, fs_start_first, fs_end_first, path_links_second,
                                    fs_start_second,fs_end_second, num_FS_ip, current_src, DC, current_dst,
                                    delay_process
        """
        # 选择DC，计算时间
        paths_record, DC, path_first, path_second, num_FS, delay_forward, delay_propagation, delay_process, current_TTL, data_for_each_VM, block_type = self.select_DC(
            current_src, current_dst, slice_piece, computing_queue, Candidate_Paths, path_id_first, path_id_second,
            current_bandwidth, current_data, delay_idea, ip_slot_map, ip_slot_map_t, slot_map, slot_map_t, FS_id_first,
            optical_slice_delay_nonsensitive, ip_slice_delay_nonsensitive, computing_slice_delay_nonsensitive,
            FS_id_second, strategy)
        return paths_record, DC, path_first, path_second, num_FS, delay_forward, delay_propagation, delay_process, current_TTL, data_for_each_VM, block_type

    def select_DC(self, current_src, current_dst, slice_piece, computing_queue, Candidate_Paths, path_id_first,
                  path_id_second, current_bandwidth, current_data, delay_idea, ip_slot_map, ip_slot_map_t, slot_map,
                  slot_map_t,
                  FS_id_first, optical_slice, ip_slice, computing_slice, FS_id_second, strategy):
        """
        暂时先不做区分，时延敏感和时延非敏感相同

        :param current_src: 起点
        :param current_dst: 终点
        :param slice_piece: 切片块编号 得到切片范围[]
        :param computing_queue: 计算队列
        :param Candidate_Paths: k条路
        :param path_id_first: 第一条路ID
        :param path_id_second: 第二条路ID
        :param current_bandwidth: 业务带宽
        :param current_data: 业务量
        :param delay_idea: 理想时延
        :param ip_slot_map:
        :param ip_slot_map_t:
        :param slot_map:
        :param slot_map_t:
        :param FS_id_first:
        :param optical_slice:
        :param ip_slice:
        :param computing_slice:
        :param FS_id_second:
        :param strategy: DC策略
        :return: 返回的DC及路径等信息
        """
        # 根据不同得到测策略选择DC
        # 判断可行性，返回时延
        paths_record = [9999999999]  # 如果失败的默认值
        DC = -1  #
        path_first = -1
        path_second = -1
        num_FS = -1
        delay_forward = -1
        delay_propagation = -1
        delay_process = -1
        current_TTL = -1

        if strategy == "load_balance":
            # 计算每个节点的VM队列长度，选择VM总长度最短的DC
            queue = [0 for x in range(self.NODE_NUM)]  # 每个节点的IP加存储综合负载
            for n in range(self.NODE_NUM):
                ql = queue[n]
                for vm in range(computing_slice[slice_piece][1] - computing_slice[slice_piece][0]):
                    # print(self.req_id, computing_slice[slice_piece])
                    ql += (computing_queue[n][computing_slice[slice_piece][0] + vm]) / self.VM_MEMORY + \
                          1 - (np.sum(ip_slot_map[n]) / ((ip_slice[slice_piece][1] -
                                                          ip_slice[slice_piece][
                                                              0]) * self.Ropt_ip))  # 计算每个节点的IP加存储综合负载 数据/总存储 + 占用IP/切片块总IP容量
                queue[n] = ql
            DC = queue.index(min(queue)) + 1  # 选择队列最短的DC节点，节点从1开始
            while DC == current_src or DC == current_dst:  # DC非起点和终点
                queue[DC - 1] = 9999999
                DC = queue.index(min(queue)) + 1
            path_first, path_second, num_FS, self.delay_forward, self.delay_propagation, self.delay_process, current_TTL, computing_flag, data_for_each_VM = self.calculate_currentTTL(
                DC, current_src, current_dst, Candidate_Paths, path_id_first, path_id_second, current_bandwidth,
                current_data, slice_piece, computing_queue, computing_slice)
            # print(self.delay_process)
            paths_record, block_type = self.judge_ability(paths_record, optical_slice, current_TTL,
                                                          delay_idea, current_bandwidth, ip_slot_map, ip_slot_map_t,
                                                          slot_map, slot_map_t, num_FS, path_first, FS_id_first,
                                                          slice_piece, current_src, DC, ip_slice,
                                                          path_second, FS_id_second, current_dst, self.delay_process,
                                                          computing_flag)

            # print(self.req_id, self.req_type, delay_forward, delay_propagation, delay_process)

        elif strategy == "time_shortest":  # 选择总时间最短的DC
            # 似乎不能提前计算
            min_paths_record = paths_record
            min_current_TTL = 9999999
            min_DC = -1
            min_path_first = -1
            min_path_second = -1
            min_num_FS = -1
            min_delay_forward = -1
            min_delay_propagation = -1
            min_delay_process = -1
            min_data_for_each_VM = []
            min_block_type = ""
            for DC in range(self.NODE_NUM):  # 计算所有DC的时间，得到时间最短minDC
                if DC + 1 != current_src and DC + 1 != current_dst:
                    DC = DC + 1
                    path_first, path_second, num_FS, self.delay_forward, self.delay_propagation, self.delay_process, current_TTL, computing_flag, data_for_each_VM = self.calculate_currentTTL(
                        DC, current_src, current_dst, Candidate_Paths, path_id_first, path_id_second, current_bandwidth,
                        current_data, slice_piece, computing_queue, computing_slice)
                    paths_record, block_type = self.judge_ability(paths_record, optical_slice, current_TTL,
                                                                  delay_idea, current_bandwidth, ip_slot_map,
                                                                  ip_slot_map_t, slot_map, slot_map_t, num_FS,
                                                                  path_first,
                                                                  FS_id_first, slice_piece, current_src, DC,
                                                                  ip_slice, path_second, FS_id_second,
                                                                  current_dst, self.delay_process, computing_flag)
                    if len(paths_record) != 1 and current_TTL < min_current_TTL:
                        min_DC = DC
                        min_current_TTL = current_TTL
                        min_paths_record = paths_record
                        min_path_first = path_first
                        min_path_second = path_second
                        min_num_FS = num_FS
                        min_delay_forward = delay_forward
                        min_delay_propagation = delay_propagation
                        min_delay_process = delay_process
                        min_data_for_each_VM = data_for_each_VM
                        min_block_type = block_type
            return min_paths_record, min_DC, min_path_first, min_path_second, min_num_FS, min_delay_forward, min_delay_propagation, min_delay_process, min_current_TTL, min_data_for_each_VM, min_block_type


        elif strategy == "random":  # 随机选择DC
            DC = np.random.randint(self.NODE_NUM) + 1  # 从1开始
            while DC == current_src or DC == current_dst:
                DC = np.random.randint(self.NODE_NUM) + 1  # 从1开始
            # 计算时间
            path_first, path_second, num_FS, self.delay_forward, self.delay_propagation, self.delay_process, current_TTL, computing_flag, data_for_each_VM = self.calculate_currentTTL(
                DC, current_src, current_dst, Candidate_Paths, path_id_first, path_id_second, current_bandwidth,
                current_data, slice_piece, computing_queue, computing_slice)
            paths_record, block_type = self.judge_ability(paths_record, optical_slice, current_TTL,
                                                          delay_idea, current_bandwidth, ip_slot_map, ip_slot_map_t,
                                                          slot_map, slot_map_t, num_FS, path_first, FS_id_first,
                                                          slice_piece, current_src, DC, ip_slice,
                                                          path_second, FS_id_second, current_dst, self.delay_process,
                                                          computing_flag)

        return paths_record, DC, path_first, path_second, num_FS, self.delay_forward, self.delay_propagation, self.delay_process, current_TTL, data_for_each_VM, block_type

    def get_process_delay(self, DC, data_for_each_VM, first_path_delay_propagation, computing_queue):
        """
                返回排队处理时间，从请求到达DC节点至所有VM都处理完数据的时间
                :param computing_queue: 计算缓存队列
                :param first_path_delay_propagation: 第一条路的传输时间
                :param DC: 选择的数据处理节点，从1开始
                :param data_for_each_VM: 每个VM 处理的数据
                :return: delay_process排队处理时间，从请求到达DC节点至所有VM都处理完数据的时间
                """
        max_process_delay = 0
        tmp = []  # 不能直接修改data_for_each_VM
        for d in data_for_each_VM:
            tmp.append(d)
        for i in range(len(tmp)):
            """
            if data_for_each_VM[i] != 0:
                data_for_each_VM[i] += computing_queue[DC - 1][i]  # 总数据 = 新数据+原有数据队列
                if first_path_delay_propagation >= computing_queue[DC - 1][i]: # 存储数据可以处理完
                    temp = 0
                else:
                    temp = first_path_delay_propagation
                if (data_for_each_VM[i] / self.vm_process_speed) - temp > max_process_delay:
                    max_process_delay = (data_for_each_VM[i] / self.vm_process_speed) - temp  # 处理时间为最大的（数据量/处理速度 - 第一段传输时间）
            """

            if tmp[i] != 0:
                if first_path_delay_propagation < computing_queue[DC - 1][
                    i]:  # 如果第一段路径时间内无法处理完所有数据，则至业务数据处理完之前所要处理的总数据为
                    tmp[i] += computing_queue[DC - 1][i]
                    tmp[i] = tmp[
                                 i] - first_path_delay_propagation  # 现加再减。如果第一段路径时间内可以处理完所有数据，仅需处理该业务的数据，及不变
                if tmp[i] / self.vm_process_speed > max_process_delay:
                    max_process_delay = tmp[i] / self.vm_process_speed

        if max_process_delay == 0:
            print("max_process_delay == 0", tmp, self.req_id, self.current_data)
        # if max_process_delay > (self.max_delay):
        #     print("get_process_delay",self.req_id, self.req_type, self.delay_process, np.sum(self.computing_queue))
        return max_process_delay

    def update_computing_for_committing(self, computing_queue, DC, data_for_each_VM):
        """
        :param computing_queue: 每个DC中每个VM的数据量队列
        :param DC: 数据中心编号，从1开始
        :param data_for_each_VM: 本次req数据各个VM分配的数据量
        :return:
        """
        for vm in range(len(data_for_each_VM)):
            if data_for_each_VM[vm] != 0:
                computing_queue[DC - 1][vm] += data_for_each_VM[vm]
        return computing_queue

    def judge_computing_ability(self, DC, current_data, path_first, computing_queue, slice_piece, computing_slice):
        """
        该方法用于判断所选DC是否可用

        :param DC: 数据中心节点
        :param current_data: 业务带宽
        :param path_first: 到达数据中心的路径
        :param computing_queue: 计算队列
        :param slice_piece: 切片编号
        :return: flag DC是否可用 1可用，0不可用。data_for_each_VM 分配给每个VM的数据量
        """
        flag = 1
        data_for_each_VM = [0 for x in range(self.VM_TOTAL)]  # 每个DC分配的数据
        path_first_len = self.cal_len(path_first, -1)  # 得到第一段路径长度
        data_processed_for_each_VM_before_req_arrive = path_first_len * self.DELAY_IN_FIBER * self.vm_process_speed  # 每个VM业务到达前可以处理多少数据
        VM_number = computing_slice[slice_piece][1] - computing_slice[slice_piece][0]  # DC节点有多少个VM
        sum_available_memory = 0  # 可用内存和
        data_req_arrive = 0
        for v in range(VM_number):
            VM = computing_slice[slice_piece][0] + v  # VM编号
            if data_processed_for_each_VM_before_req_arrive > computing_queue[DC][VM]:  # 如果业务到达时可以处理完队列所有数据, = 0
                data_req_arrive = 0
            else:
                data_req_arrive = computing_queue[DC][VM] - data_processed_for_each_VM_before_req_arrive
            sum_available_memory += (self.VM_MEMORY - data_req_arrive)
        if sum_available_memory - current_data < 0:
            flag = 0
            return flag, data_for_each_VM
        for v in range(VM_number):
            VM = computing_slice[slice_piece][0] + v  # VM编号
            if data_processed_for_each_VM_before_req_arrive > computing_queue[DC][VM]:  # 如果业务到达时可以处理完队列所有数据, = 0
                data_left = 0
            else:
                data_left = computing_queue[DC][VM] - data_processed_for_each_VM_before_req_arrive  # 剩余存储容量
            data_for_each_VM[VM] = ((
                                                self.VM_MEMORY - data_left) / sum_available_memory) * current_data  # 按照空余队列比例进行数据分配
        # print(np.sum(data_for_each_VM), current_data)
        i = []
        for d in data_for_each_VM:
            if d != 0:
                i.append(d)
        # print(self.req_id, self.req_type, len(i), np.mean(i), self.current_data)
        return flag, data_for_each_VM

    def calculate_currentTTL(self, DC, current_src, current_dst, Candidate_Paths, path_id_first, path_id_second,
                             current_bandwidth, current_data, slice_piece, computing_queue, computing_slice):
        # 计算时间
        path_first = self._get_path(current_src, DC, Candidate_Paths, path_id_first)  # 得到第K条路径
        path_second = self._get_path(DC, current_dst, Candidate_Paths, path_id_second)  # 得到第二条路的第K条路径
        path_len = self.cal_len(path_first, path_second)  # 得到总长度
        num_FS, m = self.cal_FS(current_bandwidth, path_len)  # 根据长度和带宽选择调制模式，非敏感业务first和second是相同调制方式
        # 敏感业务持续时间等于发送时间+传输时间+查表时间+处理时间
        self.delay_forward = (current_data / (num_FS * self.BFS * m))
        self.delay_propagation = path_len * self.DELAY_IN_FIBER
        self.delay_propagation = path_len * self.DELAY_IN_FIBER
        self.first_path_delay_propagation = self.cal_len(path_first, -1) * self.DELAY_IN_FIBER
        computing_flag, data_for_each_VM = self.judge_computing_ability(DC - 1, current_data, path_first,
                                                                        computing_queue,
                                                                        slice_piece, computing_slice)  # 节点从1开始
        if computing_flag != 0:
            self.delay_process = self.get_process_delay(DC, data_for_each_VM, self.first_path_delay_propagation,
                                                        computing_queue)
        else:
            self.delay_process = 0
        current_TTL = (
                    self.delay_forward + self.delay_propagation + self.delay_process + self.OEO_delay_for_delay_sensitive)
        return path_first, path_second, num_FS, self.delay_forward, self.delay_propagation, self.delay_process, current_TTL, computing_flag, data_for_each_VM

    def cal_len(self, path_first, path_second):
        path_len = 0
        for a, b in zip(path_first[:-1], path_first[1:]):
            path_len += self.linkmap[a][b][1]
        if path_second == -1:
            return path_len
        for c, d in zip(path_second[:-1], path_second[1:]):
            path_len += self.linkmap[c][d][1]
        return path_len

    def cal_FS(self, bandwidth, path_len):
        # if path_len <= 625:
        #     num_FS = math.ceil(bandwidth / (4 * 12.5)) + 1  # 1 as guard band FS
        # elif path_len <= 1250:
        #     num_FS = math.ceil(bandwidth / (3 * 12.5)) + 1
        # elif path_len <= 2500:
        #     num_FS = math.ceil(bandwidth / (2 * 12.5)) + 1
        # else:
        #     num_FS = math.ceil(bandwidth / (1 * 12.5)) + 1
        if path_len <= 240:
            num_FS = math.ceil(bandwidth / (3 * self.BFS)) + 1  # 1 as guard band FS
            m = 3
        elif path_len <= 480:
            num_FS = math.ceil(bandwidth / (2 * self.BFS)) + 1
            m = 2
        else:  # 480km以上都按BPSK
            num_FS = math.ceil(bandwidth / (1 * self.BFS)) + 1
            m = 1
        return int(num_FS), m

    def check_list(self, map):
        c = 0
        for ii in map:
            for jj in ii:
                if jj == 1:
                    c = c + 1
        return c

    def copy_list(self, map):
        res = []
        for ii in map:
            list = []
            for jj in ii:
                list.append(jj)
            res.append(list)
        return res

    def set_slice(self, optical_slice_ds, optical_slice_dns, ip_slice_ds, ip_slice_dns, computing_slice_ds,
                  computing_slice_dns):
        """

        :param optical_slice_ds: 光时延敏感
        :param optical_slice_dns:  光时延不敏感
        :param ip_slice_ds: IP时延敏感
        :param ip_slice_dns: IP时延不敏感
        :param computing_slice_ds: 计算时延敏感
        :param computing_slice_dns: 计算时延不敏感
        :return: 三种资源是否重构
        """


        #global optical_slice_delay_sensitive, optical_slice_delay_nonsensitive
        #global ip_slice_delay_sensitive, ip_slice_delay_nonsensitive
        #global computing_slice_delay_sensitive, computing_slice_delay_nonsensitive
        optical_slice_reconfig = False # 三种资源是否重构
        ip_slice_reconfig = False
        comput_slice_reconfig = False
        orig_optical_slice_delay_sensitive = self.optical_slice_delay_sensitive
        if orig_optical_slice_delay_sensitive != optical_slice_ds:
            optical_slice_reconfig = True
        self.optical_slice_delay_sensitive = optical_slice_ds  # 光时延敏感切片
        self.optical_slice_delay_nonsensitive = optical_slice_dns  # 光时延非敏感切片

        orig_ip_slice_delay_sensitive = self.ip_slice_delay_sensitive
        if orig_ip_slice_delay_sensitive != ip_slice_ds:
            ip_slice_reconfig = True
        self.ip_slice_delay_sensitive = ip_slice_ds  # IP时延敏感切片
        self.ip_slice_delay_nonsensitive = ip_slice_dns  # IP时延非敏感切片

        orig_comput_slice_delay_sensitive = self.computing_slice_delay_sensitive
        if orig_comput_slice_delay_sensitive != computing_slice_ds:
            comput_slice_reconfig = True
        self.computing_slice_delay_sensitive = computing_slice_ds  # 计算时延敏感切片
        self.computing_slice_delay_nonsensitive = computing_slice_dns  # 计算时延非敏感切片

        return optical_slice_reconfig, ip_slice_reconfig, comput_slice_reconfig

    def record_load(self):
        # 记录负载 弃用
        f = open('sensitive_req_load.txt', 'a')
        f.write(str(self.sensitive_load) + '\n')
        f.close()
        f = open('nonsensitive_req_load.txt', 'a')
        f.write(str(self.nonsensitive_load) + '\n')
        f.close()
        self.record_time = 0
        self.nonsensitive_load = 0
        self.sensitive_load = 0
        self.num_record_req_load += 1
        pass

    def record_req_times(self, ):
        # 记录业务次数
        self.sensitive_req_load.append(self.sensitive_times)
        self.nonsensitive_req_load.append(self.nonsensitive_times)
        self.record_time = 0
        self.nonsensitive_times = 0
        self.sensitive_times = 0
        self.num_record_req_load += 1
        pass

    def slice_reconfig_fix(self, slice, ratio):
        """
        切片重构，防止切片重构太极端，最小留一份资源
        :param slice: 切片
        :param ratio: 比例
        :return: 切片位置
        """
        s = 0
        ratio = float(ratio)
        if math.ceil(slice * ratio) == 0:
            s = 1
        elif math.ceil(slice * ratio) == slice:
            s = slice - 1
        else:
            s = math.ceil(slice * ratio)
        return s

    def slice_reconfig_fix_ip(self, slice, ratio):
        """
        切片重构，防止切片重构太极端，最小留一份资源
        :param slice: 切片
        :param ratio: 比例
        :return: 切片位置
        """
        s = 0
        ratio = float(ratio)
        if math.ceil(slice * ratio) <= 0:
            s = self.Ropt_ip
        elif math.ceil(slice * ratio) >= slice:
            s = slice
        else:
            s = math.ceil(slice * ratio)
        n = slice // self.Ropt_ip
        m = slice % 5
        if m > self.Ropt_ip / 2:
            s = (n + 1 ) * 5
        else:
            s = n * self.Ropt_ip
        return s

    def judge_reconfig(self, slice_now, slice_old): # 判断两个切片是否一致
        opt_config = False
        ip_config = False
        comput_config = False
        if slice_now[0] != slice_old[0]:
            opt_config = True
        if slice_now[1] != slice_old[1]:
            ip_config = True
        if slice_now[2] != slice_old[2]:
            comput_config = True

        # print("光资源距离：",slice_now[0] - slice_old[0], "ip资源距离：", slice_now[1] - slice_old[1], "缓存距离：",slice_now[2] - slice_old[2])

        return opt_config, ip_config, comput_config


    def increase_delay_due2_slice_reconfig(self, optical_slice_reconfig, ip_slice_reconfig, comput_slice_reconfig): # 重构后增加业务时延，增加资源占用时间，判断是否阻塞
        # temp_ = [ [第一条路， 第二条路]，[ [ 第一条路start fs, 第一条路end fs], [第二条路start fs, 第二条路end fs] ]  ]
        block_req = []
        vm_increased = []
        increase_time = 0
        sum_block_req_working_time_multip_data = 0  # 重构导致失败的业务已经工作的时间*数据量 之和
        # print(sum_block_req_working_time_multip_data,"1")

        # 如果都没有重构，则return
        if (optical_slice_reconfig == False) and (ip_slice_reconfig == False) and (comput_slice_reconfig == False):
            return 0, 0
        # 如果光重构了，则重构时间仅需要加上光重构时间
        if optical_slice_reconfig == True:
            increase_time += self.delay_increase_per_optical_slice_reconfig
        else:
            # 如果光没有重构
            if self.delay_increase_per_ip_slice_reconfig > self.delay_increase_per_comput_slice_reconfig:# IP重构时间大，
                if ip_slice_reconfig == True: # 光没有重构，且IP重构
                    increase_time += self.delay_increase_per_ip_slice_reconfig
                else:# 光不重构，IP不重构，计算重构
                    increase_time += self.delay_increase_per_comput_slice_reconfig
            else: # 计算重构时间长
                if comput_slice_reconfig == True: # 光没有重构，且计算重构
                    increase_time += self.delay_increase_per_comput_slice_reconfig
                else:# 光不重构，计算不重构，IP重构
                    increase_time += self.delay_increase_per_ip_slice_reconfig

        if True: # 先判断所有业务都受重构影响
            for req in self.request_set_optical.keys():
                # self.increase_delay_due2_reconfig_total += increase_time # 重构增加的总时延，尝试修改奖励
                if self.request_set_delay[req][1] + increase_time >= self.request_set_delay[req][0]:
                    block_req.append(req)
                    sum_block_req_working_time_multip_data += self.request_set_delay[req][1] *  self.requset_data[req]
                else:
                    vm_increased.append(self.increase_delay(req, vm_increased, increase_time)) # 增加资源占用时延，记录增加时延的VN
                    # 记录重构增加的时延
                    self.delay_due2_slice_reconfig += increase_time

        for req in block_req:
            self.release_recourse_of_block_req(req) # 释放阻塞业务资源
            self.num_blocks += 1
            self.block_due2_reconfig += 1
            if req in self.req_sensitive:
                self.sensitive_block_due2_reconfig += 1
            else:
                self.nonsensitive_block_due2_reconfig += 1

        return increase_time, sum_block_req_working_time_multip_data

    def increase_delay(self, req_id, vm_increased, increase_time):  # 增加时延

        # 光资源时延增加
        temp_ = self.request_set_optical[req_id]
        first_path_id = temp_[0][0]
        first_path_fs = temp_[1][0]
        second_path_id = temp_[0][1]
        second_path_fs = temp_[1][1]
        self.request_set_optical[req_id][2] += increase_time  # 根据request_set_optical判断的是否任务结束
        for link in first_path_id:
            for i in range(first_path_fs[1] - first_path_fs[0]):
                # print("req_id, link, first_path_fs[0] + i", req_id, link, first_path_fs[0] + i)
                # print("before",path, first_path_fs[0] + i, self.slot_map_t[path][first_path_fs[0] + i])
                self.slot_map_t[link][first_path_fs[0] + i] += increase_time
                # print("after",path, first_path_fs[0] + i, self.slot_map_t[path][first_path_fs[0] + i])
        for link in second_path_id:
            for i in range(second_path_fs[1] - second_path_fs[0]):
                # print("req_id, link, first_path_fs[0] + i", req_id, link, second_path_fs[0] + i, self.slot_map[link][second_path_fs[0] + i])
                self.slot_map_t[link][second_path_fs[0] + i] += increase_time

        # IP资源时延增加
        first_path_IP_slot_map = self.request_set_ip[req_id][0]
        second_path_IP_slot_map = self.request_set_ip[req_id][1]
        for n in range(len(first_path_IP_slot_map)):
            for s in range(len(first_path_IP_slot_map[0])):
                if first_path_IP_slot_map[n][s] == 0:
                    self.ip_slot_map_t[n][s] += increase_time
        for n in range(len(second_path_IP_slot_map)):
            for s in range(len(second_path_IP_slot_map[0])):
                if second_path_IP_slot_map[n][s] == 0:
                    self.ip_slot_map_t[n][s] += increase_time

        # 计算资源时延增加
        used_VM = self.request_set_computing[req_id]
        dc = used_VM[0]
        VMs = used_VM[1]
        for vm in range(len(VMs)):
            if VMs[vm] == 1:  # vm被占
                if [dc, vm] not in vm_increased:  # 重构影响设备，因此vm只需增加延迟一次
                    vm_increased.append([dc, vm])
                    self.computing_queue[dc - 1][vm] += increase_time * self.vm_process_speed
                    # dc为计算节点编码，从1开始，有dc=14，computing_queue序号从0开始，没有14号，因此需要-1

    def release_recourse_of_block_req(self, req_id):  # 释放阻塞业务资源
        # print("释放业务",req_id)
        # 光资源释放
        temp_ = self.request_set_optical[req_id]
        first_path_id = temp_[0][0]
        first_path_fs = temp_[1][0]
        second_path_id = temp_[0][1]
        second_path_fs = temp_[1][1]
        for link in first_path_id:
            for i in range(first_path_fs[1] - first_path_fs[0]):
                self.slot_map_t[link][first_path_fs[0] + i] = 0
                self.slot_map[link][first_path_fs[0] + i] = 1
        for link in second_path_id:
            for i in range(second_path_fs[1] - second_path_fs[0]):
                self.slot_map_t[link][second_path_fs[0] + i] = 0
                self.slot_map[link][second_path_fs[0] + i] = 1

        # IP资源释放
        first_path_IP_slot_map = self.request_set_ip[req_id][0]
        second_path_IP_slot_map = self.request_set_ip[req_id][1]
        for n in range(len(first_path_IP_slot_map)):
            for s in range(len(first_path_IP_slot_map[0])):
                if first_path_IP_slot_map[n][s] == 0:
                    self.ip_slot_map_t[n][s] = 0
                    self.ip_slot_map[n][s] = 1
        for n in range(len(second_path_IP_slot_map)):
            for s in range(len(second_path_IP_slot_map[0])):
                if second_path_IP_slot_map[n][s] == 0:
                    self.ip_slot_map_t[n][s] = 0
                    self.ip_slot_map[n][s] = 1

        # 计算资源释放
        used_VM = self.request_set_computing[req_id]
        dc = used_VM[0] # 该业务的dc节点
        VMs = used_VM[1] # 该业务使用的vm
        data_for_each_vm = used_VM[2] # 每个vm分配的数据
        orig_current_TTL = used_VM[3] # 原本的业务持续时间
        first_path_delay_propagation = used_VM[4] # 第一段路径传输时间
        data_unprocessed_for_each_vm = [] # 每个vm未处理的该业务数据

        time_data_processed = orig_current_TTL - first_path_delay_propagation - self.request_set_optical[req_id][2] # 每个vm已经处理该业务的时间=原本业务时间-第一条路径传输时间-现在业务剩余时间
        if time_data_processed <= 0:
            time_data_processed = 0
        # 计算每个vm未处理的该业务数据
        for vm in range(len(VMs)):
            if data_for_each_vm[vm] == 0:
                data_unprocessed_for_each_vm.append(0)
            else:
                data = data_for_each_vm[vm] - time_data_processed * self.vm_process_speed
                if data <= 0:
                    data = 0 # 重构的时刻处于一段路径传输时间，还未处理数据
                data_unprocessed_for_each_vm.append(data)




        for vm in range(len(VMs)):
            if VMs[vm] == 1:  # vm被占
                # dc为计算节点编码，从1开始，有dc=14，computing_queue序号从0开始，没有14号，因此需要-1
                # 每个阻塞业务都要释放存储资源
                # print("self.request_set_optical[req_id][2]", self.request_set_optical[req_id][2])
                # print("self.computing_queue[dc - 1][vm]", self.computing_queue[dc - 1][vm])
                data = self.computing_queue[dc - 1][vm] - data_unprocessed_for_each_vm[vm]
                # print(self.computing_queue[dc - 1][vm], data_unprocessed_for_each_vm[vm], data)

                if data == 0:
                    self.computing_queue[dc - 1][vm] = 0
                else:
                    self.computing_queue[dc - 1][vm] = data




        # 删除业务记录

        del self.request_set_optical[req_id]  # 删除所有到时间的request
        del self.request_set_ip[req_id]
        del self.request_set_computing[req_id]

        pass

    def req_info(self, req_id): # 通过业务id查看业务资源占用情况
        print("业务",req_id,"是否还存在",req_id in self.request_set_optical.keys())
        # 光资源
        if req_id in self.request_set_optical.keys():
            print("第一个path的第一个link，第一个频隙:",self.request_set_optical[req_id][0][0][0],self.request_set_optical[req_id][1][1][0])
            print("第一个path的第一个link第一个频隙的占用情况和占用剩余时间", self. slot_map[self.request_set_optical[req_id][0][0][0]][self.request_set_optical[req_id][1][1][0]], self.slot_map_t[self.request_set_optical[req_id][0][0][0]][self.request_set_optical[req_id][1][1][0]])
        # IP资源
            c = 0
            for l in self.request_set_ip[req_id][0]:
                c += l.count(0)
            print("第一条路径IP占用个数", c)
            for n in range(len(self.request_set_ip[req_id][0])):
                for s in range(len(self.request_set_ip[req_id][0][0])):
                    if self.request_set_ip[req_id][0][n][s] == 0:
                        print("第一个IP隙和占用时间",self.ip_slot_map[n][s], self.ip_slot_map_t[n][s],"节点，IP隙",n,s)
                        break
        # 计算资源
            print("计算资源dc",self.request_set_computing[req_id][0], "vm",self.request_set_computing[req_id][1])

    def calcualte_ratio(self):
        # a = (1 + math.sin(self.time / (math.pi * 8))) / 2
        # b = (1 + math.sin(self.time / (math.pi * 8))) / 2 - 0.5
        # return 0.15 + self.trapWave(self.time, 40, 0.01)
        # return (math.sin(self.time / (math.pi * 80))) /2.1 + 0.5 # sin函数分钟级别的业务动态
        # return self.get_ratio_from_nanhua(self.time)
        a = int(self.time)
        return self.yvals[a]
        return self.get_ratio_from_nanhua(self.time)


    def get_y(self, x1, y1, x2, y2, x): # 通过两个点计算一次函数
        if y1 == y2:
            y = y1
        else:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            y = a * x + b
        return y

    def get_ratio_from_nanhua(self, x): # 得到华南老师论文的业务比例
        l = [
            0.05, 0.125, 0.125, 0.3, 0.3, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.83, 0.83, 0.75, 0.75, 0.83, 0.83, 0.67, 0.67,
            0.2, 0.2, 0.1, 0.1, 0.05,
            0.05, 0.125, 0.125, 0.3, 0.3, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.83, 0.83, 0.75, 0.75, 0.83, 0.83, 0.67, 0.67,
            0.2, 0.2, 0.1, 0.1, 0.05,
            0.05, 0.125, 0.125, 0.3, 0.3, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.83, 0.83, 0.75, 0.75, 0.83, 0.83, 0.67, 0.67,
            0.2, 0.2, 0.1, 0.1, 0.05,
        ]
        time = int(x / 60)
        x1 = time
        y1 = l[x1]
        x2 = time + 1
        y2 = l[x2]
        y = self.get_y(x1, y1, x2, y2, x / 60)
        return y

    def calcualte_ratio_nanhua(self, x):
        y = []
        for a in x:
            y.append(self.get_ratio_from_nanhua(a) + 0.03)

        return y


    def H(self, x):
        return 0.5 * (np.sign(x) + 1)

    def trapWave(self, xin, width=1., slope=1.):
        x = xin % (4 * width)
        y = ((self.H(x) - self.H(x - width)) * x * slope +
             (self.H(x - width) - self.H(x - 2. * width)) * width * slope +
             (self.H(x - 2. * width) - self.H(x - 3. * width)) * (3. * width * slope - x * slope))
        return y

    def exponential_smoothing(self, alpha, s):
        s2 = np.zeros(s.shape)
        s2[0] = s[0]
        for i in range(1, len(s2)):
            s2[i] = alpha * s[i] + (1 - alpha) * s2[i - 1]
        return s2

    def prediction(self, file_name, pre_num):
        # 弃用
        alpha = .70  # 平滑系数
        data = np.loadtxt(file_name)  # 读取数据
        initial_number = data

        s_single = self.exponential_smoothing(alpha, initial_number)  # 计算一次平滑
        s_double = self.exponential_smoothing(alpha, s_single)  # 在一次平滑的基础上计算二次平滑
        a_double = 2 * s_single - s_double  # 计算二次指数严滑的a
        b_double = (alpha / (1 - alpha)) * (s_single - s_double)  # 计算二次指数严滑的b
        s_pre_double = np.zeros(s_double.shape)  # 建立预测轴
        for i in range(1, len(initial_number)):
            s_pre_double[i] = a_double[i - 1] + b_double[i - 1]
        pre_double_arr = []
        for i in range(pre_num):
            pre = a_double[-1] + b_double[-1] * (i + 1)  # 预测下一年
            pre_double_arr.append(pre)

        s_triple = self.exponential_smoothing(alpha, s_double)
        a_triple = 3 * s_single - 3 * s_double + s_triple
        b_triple = (alpha / (2 * ((1 - alpha) ** 2))) * \
                   ((6 - 5 * alpha) * s_single - 2 * ((5 - 4 * alpha) * s_double) + (4 - 3 * alpha) * s_triple)
        c_triple = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single - 2 * s_double + s_triple)
        s_pre_triple = np.zeros(s_triple.shape)
        for i in range(1, len(initial_number)):
            s_pre_triple[i] = a_triple[i - 1] + b_triple[i - 1] * 1 + c_triple[i - 1] * (1 ** 2)
        pre_triple_arr = []
        for i in range(pre_num):
            pre = a_triple[-1] + b_triple[-1] * i + c_triple[-1] * (i ** 2)
            pre_triple_arr.append(pre)

        res1 = pre_double_arr
        res2 = pre_triple_arr

        return np.mean(res1), np.mean(res2)

    def prediction_ratio_list(self, ratio_list, pre_num):
        # 弃用
        alpha = .70  # 平滑系数
        #data = np.loadtxt(file_name)  # 读取数据
        initial_number = np.array(ratio_list)

        s_single = self.exponential_smoothing(alpha, initial_number)  # 计算一次平滑
        s_double = self.exponential_smoothing(alpha, s_single)  # 在一次平滑的基础上计算二次平滑
        a_double = 2 * s_single - s_double  # 计算二次指数严滑的a
        b_double = (alpha / (1 - alpha)) * (s_single - s_double)  # 计算二次指数严滑的b
        s_pre_double = np.zeros(s_double.shape)  # 建立预测轴
        for i in range(1, len(initial_number)):
            s_pre_double[i] = a_double[i - 1] + b_double[i - 1]
        pre_double_arr = []
        for i in range(pre_num):
            pre = a_double[-1] + b_double[-1] * (i + 1)  # 预测下一年
            pre_double_arr.append(pre)

        s_triple = self.exponential_smoothing(alpha, s_double)
        a_triple = 3 * s_single - 3 * s_double + s_triple
        b_triple = (alpha / (2 * ((1 - alpha) ** 2))) * \
                   ((6 - 5 * alpha) * s_single - 2 * ((5 - 4 * alpha) * s_double) + (4 - 3 * alpha) * s_triple)
        c_triple = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single - 2 * s_double + s_triple)
        s_pre_triple = np.zeros(s_triple.shape)
        for i in range(1, len(initial_number)):
            s_pre_triple[i] = a_triple[i - 1] + b_triple[i - 1] * 1 + c_triple[i - 1] * (1 ** 2)
        pre_triple_arr = []
        for i in range(pre_num):
            pre = a_triple[-1] + b_triple[-1] * i + c_triple[-1] * (i ** 2)
            pre_triple_arr.append(pre)

        res1 = pre_double_arr
        res2 = pre_triple_arr

        return np.mean(res1), np.mean(res2)

    def prediction_ratio(self, pre_num):
        # 使用
        alpha = .70  # 平滑系数

        sensitive_data = np.loadtxt("sensitive_req_load.txt").tolist()  # 读取数据
        nonsensitive_data = np.loadtxt("nonsensitive_req_load.txt").tolist() # 读取数据
        ratio = []
        for i in range(len(sensitive_data)):
            a = sensitive_data[i] / (nonsensitive_data[i] + sensitive_data[i])
            ratio.append(a)

        # print("ratio", ratio, "self.RATIO_REQ", self.RATIO_REQ, "差", np.abs(ratio[-1] - self.RATIO_REQ))


        data = np.array(ratio)  # 读取数据
        initial_number = data

        s_single = self.exponential_smoothing(alpha, initial_number)  # 计算一次平滑
        s_double = self.exponential_smoothing(alpha, s_single)  # 在一次平滑的基础上计算二次平滑
        a_double = 2 * s_single - s_double  # 计算二次指数严滑的a
        b_double = (alpha / (1 - alpha)) * (s_single - s_double)  # 计算二次指数严滑的b
        s_pre_double = np.zeros(s_double.shape)  # 建立预测轴
        for i in range(1, len(initial_number)):
            s_pre_double[i] = a_double[i - 1] + b_double[i - 1]
        pre_double_arr = []
        for i in range(pre_num):
            pre = a_double[-1] + b_double[-1] * (i + 1)  # 预测下一年
            pre_double_arr.append(pre)

        s_triple = self.exponential_smoothing(alpha, s_double)
        a_triple = 3 * s_single - 3 * s_double + s_triple
        b_triple = (alpha / (2 * ((1 - alpha) ** 2))) * \
                   ((6 - 5 * alpha) * s_single - 2 * ((5 - 4 * alpha) * s_double) + (4 - 3 * alpha) * s_triple)
        c_triple = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single - 2 * s_double + s_triple)
        s_pre_triple = np.zeros(s_triple.shape)
        for i in range(1, len(initial_number)):
            s_pre_triple[i] = a_triple[i - 1] + b_triple[i - 1] * 1 + c_triple[i - 1] * (1 ** 2)
        pre_triple_arr = []
        for i in range(pre_num):
            pre = a_triple[-1] + b_triple[-1] * i + c_triple[-1] * (i ** 2)
            pre_triple_arr.append(pre)

        res1 = pre_double_arr
        res2 = pre_triple_arr

        return np.mean(res1), np.mean(res2)

    def prediction_ratio_DRL(self, pre_num):
        alpha = .70  # 平滑系数

        sensitive_data = self.sensitive_req_load  # 读取数据
        nonsensitive_data = self.nonsensitive_req_load # 读取数据
        ratio = []
        for i in range(len(sensitive_data)):
            a = sensitive_data[i] / (nonsensitive_data[i] + sensitive_data[i])
            ratio.append(a)

        # print("ratio", ratio, "self.RATIO_REQ", self.RATIO_REQ, "差", np.abs(ratio[-1] - self.RATIO_REQ))


        data = np.array(ratio)  # 读取数据
        initial_number = data

        s_single = self.exponential_smoothing(alpha, initial_number)  # 计算一次平滑
        s_double = self.exponential_smoothing(alpha, s_single)  # 在一次平滑的基础上计算二次平滑
        a_double = 2 * s_single - s_double  # 计算二次指数严滑的a
        b_double = (alpha / (1 - alpha)) * (s_single - s_double)  # 计算二次指数严滑的b
        s_pre_double = np.zeros(s_double.shape)  # 建立预测轴
        for i in range(1, len(initial_number)):
            s_pre_double[i] = a_double[i - 1] + b_double[i - 1]
        pre_double_arr = []
        for i in range(pre_num):
            pre = a_double[-1] + b_double[-1] * (i + 1)  # 预测下一年
            pre_double_arr.append(pre)

        s_triple = self.exponential_smoothing(alpha, s_double)
        a_triple = 3 * s_single - 3 * s_double + s_triple
        b_triple = (alpha / (2 * ((1 - alpha) ** 2))) * \
                   ((6 - 5 * alpha) * s_single - 2 * ((5 - 4 * alpha) * s_double) + (4 - 3 * alpha) * s_triple)
        c_triple = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single - 2 * s_double + s_triple)
        s_pre_triple = np.zeros(s_triple.shape)
        for i in range(1, len(initial_number)):
            s_pre_triple[i] = a_triple[i - 1] + b_triple[i - 1] * 1 + c_triple[i - 1] * (1 ** 2)
        pre_triple_arr = []
        for i in range(pre_num):
            pre = a_triple[-1] + b_triple[-1] * i + c_triple[-1] * (i ** 2)
            pre_triple_arr.append(pre)

        res1 = pre_double_arr
        res2 = pre_triple_arr
        dou = np.mean(res1)
        tri =  np.mean(res2)


        return dou, tri, ratio

    def read_recent_req_times(self):
        # 使用
        alpha = .70  # 平滑系数

        sensitive_data = np.loadtxt("sensitive_req_load.txt").tolist()  # 读取数据
        nonsensitive_data = np.loadtxt("nonsensitive_req_load.txt").tolist() # 读取数据
        ratio = []
        for i in range(len(sensitive_data)):
            a = sensitive_data[i] / (nonsensitive_data[i] + sensitive_data[i])
            ratio.append(a)
        return ratio

    def network_initialization(self):
        # global slot_map, slot_map_t, ip_slot_map, ip_slot_map_t, computing_queue, service_data, lambda_intervals, request_set_optical, request_set_ip, request_set_computing, req_id, num_blocks, num_success, delay_sum, delay_success_sum, delay_block_sum, delay_forward_sum, delay_propagation_sum, delay_process_sum, delay_idea_sum, block_first_path_optical_not_enough, block_first_path_IP_not_enough, block_first_path_all_not_enough, block_second_path_optical_not_enough, block_second_path_IP_not_enough, block_second_path_all_not_enough, block_over_time, block_computing_not_enough, time_to, num_req_measure, resource_util_opt, resource_util_ip, resource_util_memory
        self.slot_map = [[1 for x in range(self.OPTICAL_SLOT_TOTAL)] for y in
                         range(self.LINK_NUM)]  # Initialized to be all available 都可用
        self.slot_map_t = [[0 for x in range(self.OPTICAL_SLOT_TOTAL)] for y in
                           range(self.LINK_NUM)]  # the time each FS will be occupied 每个频隙将占用的时间
        self.ip_slot_map = [[1 for x in range(self.IP_SLOT_TOTAL)] for y in range(self.NODE_NUM)]  # IP资源 都可用 注意多一个1
        self.ip_slot_map_t = [[0 for x in range(self.IP_SLOT_TOTAL)] for y in range(self.NODE_NUM)]  # IP频隙将占用的时间 都0
        self.computing_queue = [[0 for x in range(self.VM_TOTAL)] for y in range(self.NODE_NUM)]  # 每各DC排队情况 从0开始
        self.service_data = self.lambda_data[np.random.randint(0, self.len_lambda_data)]  # 随机选一个数据量
        self.lambda_intervals = 1 / self.LAMBDA_REQ  # average time interval between request 业务时间间隔
        self.requset_data = {} # 记录每个业务的带宽
        self.request_set_optical = {}  # 记录每各请求光资源占用情况
        self.request_set_ip = {}  # 记录每个请求的IP隙占用情况[ 请求id: [node_num个len=IPSLOT_NUM的数组] ]
        self.request_set_computing = {} # [ 请求id: [DC VM ] ]

        self.request_set_delay = {} # 记录成功业务的最大容忍时延和真实时延，用于判断重构之后业务是否阻塞

        self.req_id = 0
        self.num_blocks = 0  # 3000次之后总阻塞数
        self.num_success = 0
        self.delay_sum = 0  # 3000次之后总延迟
        self.delay_idea_sum = 0 # 记录成功业务的idea delay，用于计算reward
        self.delay_success_sum = 0  #
        self.delay_block_sum = 0
        self.delay_forward_sum = 0
        self.delay_propagation_sum = 0
        self.delay_due2_slice_reconfig = 0 # 重构产生的延迟
        self.first_path_delay_propagation = 0 # 业务第一条路径传输时间，用于重构阻塞后计算资源释放
        self.delay_process_sum = 0
        # self.delay_idea_sum = 0  # 调参用
        self.block_first_path_optical_not_enough = 0  # 阻塞原因
        self.block_first_path_IP_not_enough = 0
        self.block_first_path_all_not_enough = 0
        self.block_second_path_optical_not_enough = 0  # 阻塞原因
        self.block_second_path_IP_not_enough = 0
        self.block_second_path_all_not_enough = 0
        self.block_over_time = 0
        self.block_computing_not_enough = 0
        self.block_due2_reconfig = 0 # 由于重构造成的阻塞
        self.time_to = 0
        self.num_req_measure = self.NUM_REQ_MEASURE
        self.resource_util_opt = []
        self.resource_util_ip = []
        self.resource_util_memory = []

        self.sensitive_block = 0
        self.nonsensitive_block = 0
        self.sensitive_block_due2_reconfig = 0
        self.nonsensitive_block_due2_reconfig = 0
        self.req_num_sensitive = 0
        self.req_num_nonsensitive = 0

    def request_initialization(self):
        # 释放到期业务资源
        # 对于到时请求，释放slot_map资源，清除对应request_set中内容。
        # 对于非到时请求，slot_map_t中占用的资源时间减去time_to
        # self.block = 0  # 当前业务阻塞
        # self.delay = 0  # 当前业务延迟
        self.delay_block = 0  # 当前业务阻塞/延迟太长产生的延迟
        self.delay_forward = 0  # 当前业务发送延迟
        self.delay_propagation = 0  # 当前业务传输延迟
        self.delay_process = 0  # 当前业务处理延迟
        self.time_to = 0  # 每time_to时间间隔发生一次业务
        self.req_sensitive = [] # 记录敏感业务
        self.req_nonsensitive = []
        while self.time_to == 0:
            self.time_to = np.random.exponential(self.lambda_intervals)  # 业务间隔时间服从指数分布
        # generate current request
        if self.nonuniform is True:  # 非均匀业务
            sd_onehot = [x for x in range(self.num_src_dest_pair)]
            sd_id = np.random.choice(sd_onehot, p=self.prob_arr)
            temp = self.Src_Dest_Pair[sd_id]
        else:  # 均匀业务
            temp = self.Src_Dest_Pair[np.random.randint(0, self.num_src_dest_pair)]  # 随机选择节点对
        self.current_src = temp[0]
        self.current_dst = temp[1]
        self.current_bandwidth = np.random.randint(self.bandwidth[0], self.bandwidth[1])  # 随机带宽
        self.current_data = 0
        self.delay_idea = 0
        while self.delay_idea == 0:
            self.delay_idea = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.current_data = self.delay_idea * self.current_bandwidth   # lambda_time单位是分钟，带宽是Gb/分钟

if __name__ == "__main__":
    NetworkEnv = NetworkEnv()
    print(NetworkEnv.reset())
    print("step")
    NetworkEnv.step(0)
    #print(NetworkEnv.step(0,[2, 2, 2, 0.8, 0.8, 0.8, 0.876]))
