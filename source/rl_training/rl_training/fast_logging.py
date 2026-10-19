"""Keep TensorBoard scalars while avoiding per-field CUDA sync/concatenation."""
from __future__ import annotations
import statistics
import torch


def episode_means(rows):
    """Match RSL's concatenated means, with one packed GPU-to-host transfer."""
    if not rows:
        return {}
    parts, spans, host, count = [], {}, {}, {}
    cursor = 0
    for key in rows[0]:
        spans[key], host[key], count[key] = [], 0., 0
        for row in rows:
            if key not in row:
                continue
            value = torch.as_tensor(row[key]).detach().reshape(-1)
            count[key] += value.numel()
            if value.is_cuda:
                parts.append(value)
                spans[key].append((cursor, cursor+value.numel()))
                cursor += value.numel()
            else:
                host[key] += float(value.float().sum())
    packed = torch.cat(parts).float().cpu().tolist() if parts else []
    return {key: (host[key]+sum(sum(packed[a:b]) for a,b in spans[key]))/count[key]
            for key in spans if count[key]}


def write_iteration(logger, *, it, collect_time, learn_time, loss_dict, learning_rate, action_std, **kwargs):
    if logger.writer is None:
        return
    size = logger.cfg['num_steps_per_env']*logger.num_envs*logger.gpu_world_size
    logger.tot_timesteps += size
    logger.tot_time += collect_time+learn_time
    metrics = episode_means(logger.ep_extras)
    for key, value in metrics.items():
        logger.writer.add_scalar(key if '/' in key else 'Episode/'+key, value, it)
    for key, value in loss_dict.items():
        logger.writer.add_scalar('Loss/'+key, value, it)
    logger.writer.add_scalar('Loss/learning_rate', learning_rate, it)
    logger.writer.add_scalar('Policy/mean_std', float(action_std.mean()), it)
    logger.writer.add_scalar('Perf/total_fps', int(size/(collect_time+learn_time)), it)
    logger.writer.add_scalar('Perf/collection_time', collect_time, it)
    logger.writer.add_scalar('Perf/learning_time', learn_time, it)
    if logger.rewbuffer:
        reward, length = statistics.mean(logger.rewbuffer), statistics.mean(logger.lenbuffer)
        logger.writer.add_scalar('Train/mean_reward', reward, it)
        logger.writer.add_scalar('Train/mean_episode_length', length, it)
        if logger.logger_type != 'wandb':
            logger.writer.add_scalar('Train/mean_reward/time', reward, int(logger.tot_time))
            logger.writer.add_scalar('Train/mean_episode_length/time', length, int(logger.tot_time))
    logger.ep_extras.clear()
