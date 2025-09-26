import time
import os
from collections import deque
import statistics

import torch
from rsl_rl.algorithms import PPO, PPODreamWaQ
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticDreamWaQ
from rsl_rl.env import VecEnv
from rsl_rl.utils import tensor_summary

import wandb


class OnPolicyRunnerDreamWaQ:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        runner_debug_cfg = self.cfg.get("debug")
        global_debug_cfg = train_cfg.get("debug")
        env_debug_flag = bool(int(os.getenv("RSL_RL_DEBUG", "0")))
        if runner_debug_cfg is not None:
            resolved_debug = bool(runner_debug_cfg)
        elif global_debug_cfg is not None:
            resolved_debug = bool(global_debug_cfg)
        else:
            resolved_debug = True
        self.debug_enabled = env_debug_flag or resolved_debug

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        policy_kwargs = dict(self.policy_cfg)
        policy_kwargs.setdefault("debug", self.debug_enabled)
        actor_critic: ActorCriticDreamWaQ = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            self.env.history_len,
            **policy_kwargs
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        alg_kwargs = dict(self.alg_cfg)
        alg_kwargs.setdefault("debug_enabled", self.debug_enabled)
        self.alg: PPODreamWaQ = alg_class(actor_critic, device=self.device, **alg_kwargs)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, self.env.history_len, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])
        self._debug("Storage initialized",
                    num_envs=self.env.num_envs,
                    num_steps_per_env=self.num_steps_per_env,
                    history_len=self.env.history_len,
                    actor_obs_shape=self.env.num_obs,
                    privileged_obs_shape=self.env.num_privileged_obs,
                    action_dim=self.env.num_actions)

        # Log
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
    
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # wandb init
        wandb.init(
            project="leggedgym_project",
            name=self.cfg.get("exp_name", "a1_dreamwaq_cenet_test"),
            config=train_cfg
        )
        self._debug("wandb initialized",
                    project="leggedgym_project",
                    run_name=self.cfg.get("exp_name", "a1_dreamwaq_cenet_test"))

        reset_result = self.env.reset()
        self._debug("Environment reset", reset_output=reset_result)

        self._debug("Runner initialized",
                    device=self.device,
                    log_dir=self.log_dir,
                    debug_enabled=self.debug_enabled,
                    runner_cfg=self.cfg,
                    alg_cfg=self.alg_cfg,
                    policy_cfg=self.policy_cfg)

        # print("[DEBUG] Using OnPolicyRunnerDreamWaQ")
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._debug("Learn invoked",
                    num_learning_iterations=num_learning_iterations,
                    init_at_random_ep_len=init_at_random_ep_len,
                    current_iteration=self.current_learning_iteration)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
            self._debug("Initialized random episode lengths",
                        episode_length_buf=self.env.episode_length_buf,
                        max_episode_length=self.env.max_episode_length)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        history_obs = self.env.get_history_observations()
        velocity_targets = self.env.get_velocity_targets()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, history_obs, velocity_targets = obs.to(self.device), critic_obs.to(self.device), history_obs.to(self.device), velocity_targets.to(self.device)

        self._debug("Initial buffers prepared",
                    obs=obs,
                    critic_obs=critic_obs,
                    history_obs=history_obs,
                    velocity_targets=velocity_targets)
        
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self._debug("Actor-critic set to train mode")

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self._debug("Iteration start",
                        iteration=it,
                        total_iterations=tot_iter,
                        obs=obs,
                        critic_obs=critic_obs,
                        history_obs=history_obs,
                        velocity_targets=velocity_targets)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    self._debug("Rollout step",
                                iteration=it,
                                step=i,
                                obs=obs,
                                critic_obs=critic_obs,
                                history_obs=history_obs,
                                velocity_targets=velocity_targets)
                    actions = self.alg.act(obs, critic_obs, history_obs, velocity_targets)
                    self._debug("Actions sampled",
                                iteration=it,
                                step=i,
                                actions=actions,
                                action_mean=self.alg.actor_critic.action_mean,
                                action_std=self.alg.actor_critic.action_std)
                    obs, privileged_obs, history_obs, velocity_targets, rewards, dones, infos = self.env.step(actions)
                    self._debug("Env step completed",
                                iteration=it,
                                step=i,
                                rewards=rewards,
                                dones=dones,
                                infos_keys=list(infos.keys()))
                    # update validated
                    # print("history_obs_batch[0, 0, :10]:", history_obs[0, 0, :10])  # 첫 샘플의 가장 최근 history 앞 10개
                    # print("history_obs_batch[0, -1, :10]:", history_obs[0, -1, :10])  # 같은 샘플의 가장 오래된 history 앞 10개
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, history_obs, velocity_targets, rewards, dones = obs.to(self.device), critic_obs.to(self.device), history_obs.to(self.device), velocity_targets.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self._debug("Tensors moved to device",
                                iteration=it,
                                step=i,
                                obs=obs,
                                critic_obs=critic_obs,
                                history_obs=history_obs,
                                velocity_targets=velocity_targets,
                                rewards=rewards,
                                dones=dones)
                    self.alg.process_env_step(rewards, dones, infos)
                    self._debug("Processed env step",
                                iteration=it,
                                step=i,
                                storage_step=getattr(self.alg.storage, "step", None))
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                self._debug("Rollout complete",
                            iteration=it,
                            collection_time=collection_time,
                            storage_filled_steps=self.alg.storage.step)

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                self._debug("Computed returns",
                            iteration=it,
                            critic_obs=critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_velocity_loss, mean_recon_loss, mean_kl_loss, mean_ce_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self._debug("Update complete",
                        iteration=it,
                        learn_time=learn_time,
                        mean_value_loss=mean_value_loss,
                        mean_surrogate_loss=mean_surrogate_loss,
                        mean_velocity_loss=mean_velocity_loss,
                        mean_recon_loss=mean_recon_loss,
                        mean_kl_loss=mean_kl_loss,
                        mean_ce_loss=mean_ce_loss)
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        self._debug("Learning session complete",
                    total_iterations=num_learning_iterations,
                    current_learning_iteration=self.current_learning_iteration)

        wandb.finish()
        self._debug("wandb run finished")

    def log(self, locs):
            rewbuffer = locs['rewbuffer']
            lenbuffer = locs['lenbuffer']
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += locs['collection_time'] + locs['learn_time']
            fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
            mean_std = self.alg.actor_critic.std.mean()

            log_dict = {
                "iteration": locs['it'],
                "timesteps": self.tot_timesteps,
                "Loss/value_function": locs['mean_value_loss'],
                "Loss/surrogate": locs['mean_surrogate_loss'],
                "Loss/velocity": locs['mean_velocity_loss'],
                "Loss/reconstruction": locs['mean_recon_loss'],
                "Loss/kl": locs['mean_kl_loss'],
                "Loss/ce": locs['mean_ce_loss'],
                "Loss/learning_rate": self.alg.learning_rate,
                "Policy/mean_noise_std": mean_std.item(),
                "Perf/fps": fps,
                "Perf/collection_time": locs['collection_time'],
                "Perf/learning_time": locs['learn_time']
            }

            if len(rewbuffer) > 0:
                log_dict["Train/mean_reward"] = statistics.mean(rewbuffer)
                log_dict["Train/mean_episode_length"] = statistics.mean(lenbuffer)

            self._debug("Logging metrics",
                        iteration=locs['it'],
                        log_dict=log_dict,
                        total_timesteps=self.tot_timesteps,
                        total_time=self.tot_time)

            wandb.log(log_dict)

    # def log(self, locs, width=80, pad=35):
    #     self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
    #     self.tot_time += locs['collection_time'] + locs['learn_time']
    #     iteration_time = locs['collection_time'] + locs['learn_time']

    #     wandb.log({
    #         "Loss/value_function": locs['mean_value_loss'],
    #         "Loss/surrogate": locs['mean_surrogate_loss'],
    #         "Loss/velocity": locs['mean_velocity_loss'],
    #         "Loss/reconstruction": locs['mean_recon_loss'],
    #         "Loss/kl": locs['mean_kl_loss'],
    #         "Loss/ce": locs['mean_ce_loss'],
    #         "Loss/learning_rate": self.alg.learning_rate,
    #         "Policy/mean_noise_std": self.alg.actor_critic.std.mean().item(),
    #         "Perf/total_fps": int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time'])),
    #         "Perf/collection_time": locs['collection_time'],
    #         "Perf/learning_time": locs['learn_time'],
    #         "Train/mean_reward": statistics.mean(locs['rewbuffer']) if len(locs['rewbuffer']) > 0 else 0,
    #         "Train/mean_episode_length": statistics.mean(locs['lenbuffer']) if len(locs['lenbuffer']) > 0 else 0,
    #         "iteration" : locs['it'],
    #         "total_timesteps": self.tot_timesteps,
    #         "total_time": self.tot_time
    #     })

    #     ep_string = f''
    #     if locs['ep_infos']:
    #         for key in locs['ep_infos'][0]:
    #             infotensor = torch.tensor([], device=self.device)
    #             for ep_info in locs['ep_infos']:
    #                 # handle scalar and zero dimensional tensor infos
    #                 if not isinstance(ep_info[key], torch.Tensor):
    #                     ep_info[key] = torch.Tensor([ep_info[key]])
    #                 if len(ep_info[key].shape) == 0:
    #                     ep_info[key] = ep_info[key].unsqueeze(0)
    #                 infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
    #             value = torch.mean(infotensor)
    #             self.writer.add_scalar('Episode/' + key, value, locs['it'])
    #             ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    #     mean_std = self.alg.actor_critic.std.mean()
    #     fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

    #     self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
    #     self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
    #     self.writer.add_scalar('Loss/velocity', locs['mean_velocity_loss'], locs['it'])
    #     self.writer.add_scalar('Loss/reconstruction', locs['mean_recon_loss'], locs['it'])
    #     self.writer.add_scalar('Loss/kl', locs['mean_kl_loss'], locs['it'])
    #     self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
    #     self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
    #     self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
    #     self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
    #     self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
    #     if len(locs['rewbuffer']) > 0:
    #         self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
    #         self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
    #         self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
    #         self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

    #     str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

    #     if len(locs['rewbuffer']) > 0:
    #         log_string = (f"""{'#' * width}\n"""
    #                       f"""{str.center(width, ' ')}\n\n"""
    #                       f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
    #                         'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
    #                       f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
    #                       f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
    #                       f"""{'Velocity loss:':>{pad}} {locs['mean_velocity_loss']:.4f}\n"""
    #                       f"""{'Reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
    #                       f"""{'KL loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
    #                       f"""{'CE loss:':>{pad}} {locs['mean_ce_loss']:.4f}\n"""
    #                       f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
    #                       f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
    #                       f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
    #                     #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
    #                     #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
    #     else:
    #         log_string = (f"""{'#' * width}\n"""
    #                       f"""{str.center(width, ' ')}\n\n"""
    #                       f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
    #                         'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
    #                       f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
    #                       f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
    #                       f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
    #                     #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
    #                     #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

    #     log_string += ep_string
    #     log_string += (f"""{'-' * width}\n"""
    #                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
    #                    f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
    #                    f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
    #                    f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
    #                            locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
    #     print(log_string)

    def save(self, path, infos=None):
        self._debug("Saving checkpoint", path=path, infos=infos, iteration=self.current_learning_iteration)
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        self._debug("Loading checkpoint", path=path, load_optimizer=load_optimizer)
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        self._debug("Checkpoint loaded",
                    path=path,
                    iteration=self.current_learning_iteration,
                    contains_infos=loaded_dict.get('infos') is not None)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self._debug("Fetching inference policy", target_device=device)
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
            self._debug("Moved actor critic for inference", device=device)
        return self.alg.actor_critic.act_inference

    def _debug(self, message, **values):
        if not self.debug_enabled:
            return
        details = [f"[DreamWaQ:Runner] {message}"]
        for name, value in values.items():
            details.append(tensor_summary(name, value))
        print(" | ".join(details))
