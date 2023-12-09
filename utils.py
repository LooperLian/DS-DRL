"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, reward_queue, name, block_rate, block_queue,
           output_block, output_block_queue,
           output_block_due2_overtime, output_block_due2_overtime_queue,
           output_block_due2_reconfig, output_block_due2_reconfig_queue,
           output_block_due2_memory, output_block_due2_memory_queue,
           output_block_due2_opt_only, output_block_due2_opt_only_queue,
           output_block_due2_IP_only, output_block_due2_IP_only_queue,
           output_block_due2_optIP, output_block_due2_optIP_queue,
           output_delay, output_delay_queue,
           output_delay_due2_reconfig, output_delay_due2_reconfig_queue,
           output_delay_forward, output_delay_forward_queue,
           output_delay_propagation, output_delay_propagation_queue,
           output_delay_process, output_delay_process_queue,
           output_opt_util, output_opt_util_queue,
           output_ip_util, output_ip_util_queue,
           output_memory_util, output_memory_util_queue,
           optical_reconfig_times, optical_reconfig_times_queue,
           ip_reconfig_times, ip_reconfig_times_queue,
           comput_reconfig_times, comput_reconfig_times_queue
           ):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    reward_queue.put(global_ep_r.value)
    """
    reward_queue.put(global_ep_r.value)
    block_queue.put(block_rate)

    output_block_queue.put(output_block)
    output_block_due2_overtime_queue.put(output_block_due2_overtime)
    output_block_due2_reconfig_queue.put(output_block_due2_reconfig)
    output_block_due2_memory_queue.put(output_block_due2_memory)
    output_block_due2_opt_only_queue.put(output_block_due2_opt_only)
    output_block_due2_IP_only_queue.put(output_block_due2_IP_only)
    output_block_due2_optIP_queue.put(output_block_due2_optIP)
    output_delay_queue.put(output_delay)
    output_delay_due2_reconfig_queue.put(output_delay_due2_reconfig)
    output_delay_forward_queue.put(output_delay_forward)
    output_delay_propagation_queue.put(output_delay_propagation)
    output_delay_process_queue.put(output_delay_process)
    output_opt_util_queue.put(output_opt_util)
    output_ip_util_queue.put(output_ip_util)
    output_memory_util_queue.put(output_memory_util)
    optical_reconfig_times_queue.put(optical_reconfig_times)
    ip_reconfig_times_queue.put(ip_reconfig_times)
    comput_reconfig_times_queue.put(comput_reconfig_times)
    """


    print(
        name,
        # "Ep:", global_ep.value,
        # "| Ep_r: %.4f" % global_ep_r.value,
        # "| BLOCK RATE: %.4f" % block_rate,
        # "| output_block_due2_overtime: %.4f" % output_block_due2_overtime,
        #
        # "| output_block_due2_reconfig: %.4f" % output_block_due2_reconfig,
        # "| output_block_due2_memory: %.4f" % output_block_due2_memory,
        # "| output_block_due2_opt_only: %.4f" % output_block_due2_opt_only,
        # "| output_block_due2_IP_only: %.4f" % output_block_due2_IP_only,
        # "| output_block_due2_optIP: %.4f" % output_block_due2_optIP,
        #
        # "| OPTICAL RECONFIG TIMES: %.4f" % optical_reconfig_times,
        # "| IP RECONFIG TIMES: %.4f" % ip_reconfig_times,
        # "| COMPUT RECONFIG TIMES: %.4f" % comput_reconfig_times,

        "Ep:", global_ep.value,
        "Ep_r: %.4f" % global_ep_r.value,
        "BLOCK: %.4f" % block_rate,




    )
