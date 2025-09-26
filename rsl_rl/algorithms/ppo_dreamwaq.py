import os

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic, ActorCriticDreamWaQ
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import tensor_summary

from collections import deque

class PPODreamWaQ:
    actor_critic: ActorCriticDreamWaQ
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=48,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-4,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 beta_coef=1.0,
                 debug_enabled=True,
                 ):

        self.device = device
        self.debug_enabled = bool(debug_enabled) or bool(int(os.getenv("RSL_RL_DEBUG", "0")))

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parametersa
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # CENet parameters
        self.beta_coef = beta_coef

        self.episode_reward_window = deque(maxlen=100)

        self._debug(
            "PPODreamWaQ initialized",
            device=device,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            beta_coef=beta_coef,
        )

    def init_storage(self, num_envs, num_transitions_per_env, history_len, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            history_len,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
            debug_enabled=self.debug_enabled
        )
        self._debug("Storage initialized",
                    num_envs=num_envs,
                    num_transitions_per_env=num_transitions_per_env,
                    history_len=history_len,
                    actor_obs_shape=actor_obs_shape,
                    critic_obs_shape=critic_obs_shape,
                    action_shape=action_shape)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, history_obs, velocity_targets):
        self._debug("Act called",
                    obs=obs,
                    critic_obs=critic_obs,
                    history_obs=history_obs,
                    velocity_targets=velocity_targets)
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
            self._debug("Captured hidden states",
                        hidden_states=self.transition.hidden_states)
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, history_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.history_observations = history_obs
        self.transition.critic_observations = critic_obs
        self.transition.velocity_targets = velocity_targets
        self._debug("Transition recorded",
                    actions=self.transition.actions,
                    values=self.transition.values,
                    actions_log_prob=self.transition.actions_log_prob,
                    action_mean=self.transition.action_mean,
                    action_sigma=self.transition.action_sigma)
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self._debug("Process env step",
                    rewards=rewards,
                    dones=dones,
                    infos_keys=list(infos.keys()) if isinstance(infos, dict) else infos)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self._debug("Applied time-out bootstrapping",
                        adjusted_rewards=self.transition.rewards)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self._debug("Transition stored",
                    storage_step=self.storage.step,
                    rewards=self.transition.rewards,
                    dones=self.transition.dones)
        self.transition.clear()
        self._debug("Transition cleared")
        self.actor_critic.reset(dones)
        self._debug("Actor critic reset", dones=dones)
    
    def compute_returns(self, last_critic_obs):
        self._debug("Compute returns called", last_critic_obs=last_critic_obs)
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self._debug("Last values computed", last_values=last_values)
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        self._debug("Returns computed",
                    gamma=self.gamma,
                    lam=self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        # CENet
        mean_velocity_loss = 0
        mean_recon_loss = 0
        mean_kl_loss = 0
        mean_ce_loss = 0

        self._debug("Update started",
                    is_recurrent=self.actor_critic.is_recurrent,
                    num_learning_epochs=self.num_learning_epochs,
                    num_mini_batches=self.num_mini_batches)

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        batch_idx = 0
        for obs_batch, critic_obs_batch, history_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, velocity_targets_batch in generator:
                batch_idx += 1
                self._debug("Mini-batch fetched",
                            batch_index=batch_idx,
                            obs_batch=obs_batch,
                            critic_obs_batch=critic_obs_batch,
                            history_obs_batch=history_obs_batch,
                            actions_batch=actions_batch,
                            target_values_batch=target_values_batch,
                            advantages_batch=advantages_batch,
                            returns_batch=returns_batch,
                            old_actions_log_prob_batch=old_actions_log_prob_batch,
                            old_mu_batch=old_mu_batch,
                            old_sigma_batch=old_sigma_batch,
                            masks_batch=masks_batch,
                            velocity_targets_batch=velocity_targets_batch)
            
                # predicted_velocity_batch, _, reconstructed_obs_batch, latent_mu, logvar = self.actor_critic.forward(history_obs_batch)

                # # Problem: history_obs_batch  -> fixed
                # print("obs_batch:", obs_batch[0, :10])
                # print("history_obs_batch[0, 0, :10]:", history_obs_batch[0, 0, :10])  # 첫 샘플의 가장 최근 history 앞 10개
                # print("history_obs_batch[0, -1, :10]:", history_obs_batch[0, -1, :10])  # 같은 샘플의 가장 오래된 history 앞 10개
                self.actor_critic.act(obs_batch, history_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                predicted_velocity_batch, _, reconstructed_obs_batch, latent_mu, logvar = self.actor_critic.forward(history_obs_batch)
                self._debug("Forward results",
                            batch_index=batch_idx,
                            predicted_velocity_batch=predicted_velocity_batch,
                            reconstructed_obs_batch=reconstructed_obs_batch,
                            latent_mu=latent_mu,
                            logvar=logvar)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate
                        self._debug("Adaptive LR adjustment",
                                    batch_index=batch_idx,
                                    kl_mean=kl_mean,
                                    updated_learning_rate=self.learning_rate)


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                self._debug("Surrogate computed",
                            batch_index=batch_idx,
                            ratio=ratio,
                            surrogate=surrogate,
                            surrogate_loss=surrogate_loss)

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # CENet loss                
                velocity_loss = (predicted_velocity_batch - velocity_targets_batch).pow(2).mean()
                recon_loss = (obs_batch - reconstructed_obs_batch).pow(2).mean()

                kl_loss = -0.5 * torch.mean(1 + logvar - latent_mu.pow(2) - logvar.exp())
                ce_loss = velocity_loss + recon_loss + kl_loss * self.beta_coef

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + ce_loss
                self._debug("Loss components",
                            batch_index=batch_idx,
                            value_loss=value_loss,
                            surrogate_loss=surrogate_loss,
                            entropy=entropy_batch,
                            velocity_loss=velocity_loss,
                            recon_loss=recon_loss,
                            kl_loss=kl_loss,
                            ce_loss=ce_loss,
                            total_loss=loss)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self._debug("Optimizer step",
                            batch_index=batch_idx,
                            grad_norm=self.max_grad_norm,
                            learning_rate=self.learning_rate)

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_velocity_loss += velocity_loss.item()
                mean_ce_loss += ce_loss.item()
                mean_recon_loss += recon_loss.item()
                mean_kl_loss += kl_loss.item()
                self._debug("Accumulated losses",
                            batch_index=batch_idx,
                            mean_value_loss=mean_value_loss,
                            mean_surrogate_loss=mean_surrogate_loss,
                            mean_velocity_loss=mean_velocity_loss,
                            mean_ce_loss=mean_ce_loss,
                            mean_recon_loss=mean_recon_loss,
                            mean_kl_loss=mean_kl_loss)
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_velocity_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_kl_loss /= num_updates
        mean_ce_loss /= num_updates

        self.storage.clear()
        self._debug("Update finished",
                    num_updates=num_updates,
                    mean_value_loss=mean_value_loss,
                    mean_surrogate_loss=mean_surrogate_loss,
                    mean_velocity_loss=mean_velocity_loss,
                    mean_recon_loss=mean_recon_loss,
                    mean_kl_loss=mean_kl_loss,
                    mean_ce_loss=mean_ce_loss)

        self._debug("Storage cleared after update")

        return mean_value_loss, mean_surrogate_loss, mean_velocity_loss, mean_recon_loss, mean_kl_loss, mean_ce_loss

    def _debug(self, message, **values):
        if not self.debug_enabled:
            return
        details = [f"[DreamWaQ:PPO] {message}"]
        for name, value in values.items():
            details.append(tensor_summary(name, value))
        print(" | ".join(details))
