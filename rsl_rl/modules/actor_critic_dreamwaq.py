import os
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from rsl_rl.utils import tensor_summary

class ActorCriticDreamWaQ(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        history_len,
                        velocity_dims=3,
                        latent_dims=16,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        debug=True,
                        **kwargs):
        if kwargs:
            print("ActorCriticDreamWaQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDreamWaQ, self).__init__()

        self.debug_enabled = debug or bool(int(os.getenv("RSL_RL_DEBUG", "0")))

        activation = get_activation(activation)

        # CENet
        cenet_input_dim = num_actor_obs * (history_len + 1)

        # Encoder
        self.cenet_encoder = nn.Sequential(
            nn.Linear(cenet_input_dim, 128),
            activation,
            nn.Linear(128, 64),
            activation,
        )

        # Velocity head
        self.velocity_mu_head = nn.Linear(64, velocity_dims)
        self.velocity_logvar_head = nn.Linear(64, velocity_dims)

        # Latent head (mu, logvar)
        self.latent_mu_head = nn.Linear(64, latent_dims)
        self.latent_logvar_head = nn.Linear(64, latent_dims)

        # Decoder
        self.cenet_decoder = nn.Sequential(
            nn.Linear(latent_dims + velocity_dims, 64),
            activation,
            nn.Linear(64, 128),
            activation,
            nn.Linear(128, num_actor_obs)
        )

        self._debug(
            "Initialized CENet",
            cenet_encoder=self.cenet_encoder,
            cenet_decoder=self.cenet_decoder,
            history_len=history_len,
            velocity_dims=velocity_dims,
            latent_dims=latent_dims,
        )

        mlp_input_dim_a = num_actor_obs + velocity_dims + latent_dims
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        self._debug("Initialized Actor/Critic", actor_mlp=self.actor, critic_mlp=self.critic)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        self._debug("Reset called", dones=dones)

    def forward(self, history_observations):
        batch_size = history_observations.shape[0]
        history_observations_flat = history_observations.view(batch_size, -1)
        encoded = self.cenet_encoder(history_observations_flat)

        velocity_mu = self.velocity_mu_head(encoded)
        velocity_logvar = self.velocity_logvar_head(encoded)
        latent_mu = self.latent_mu_head(encoded)
        latent_logvar = self.latent_logvar_head(encoded)

        velocity_std = torch.exp(0.5 * velocity_logvar)
        eps_v = torch.randn_like(velocity_std)
        v = velocity_mu + eps_v * velocity_std

        latent_std = torch.exp(0.5 * latent_logvar)
        eps_z = torch.randn_like(latent_std)
        z = latent_mu + eps_z * latent_std

        reconstructed_obs = self.cenet_decoder(torch.cat([v, z], dim=-1))

        self._debug(
            "Forward pass",
            history=history_observations,
            encoded=encoded,
            velocity_mu=velocity_mu,
            velocity_std=velocity_std,
            latent_mu=latent_mu,
            latent_std=latent_std,
            reconstructed_obs=reconstructed_obs,
        )

        return v, z, reconstructed_obs, latent_mu, latent_logvar

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_input):
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean*0. + self.std)
        self._debug("Updated distribution",
                    actor_input=actor_input,
                    mean=mean,
                    std=self.std)

    def act(self, observations, history_observations, **kwargs):

        v, z, _, _, _ = self.forward(history_observations)
        actor_input = torch.cat([observations, v, z], dim=-1)

        self.update_distribution(actor_input)
        action_sample = self.distribution.sample()
        self._debug("Act",
                    observations=observations,
                    history=history_observations,
                    actor_input=actor_input,
                    action_sample=action_sample)
        return action_sample
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history_observations):
        predicted_velocity, z, _, _, _ = self.forward(history_observations)
        actor_input = torch.cat([observations, predicted_velocity, z], dim=-1)
        actions_mean = self.actor(actor_input)
        self._debug("Act inference",
                    observations=observations,
                    predicted_velocity=predicted_velocity,
                    latent=z,
                    actor_input=actor_input,
                    actions_mean=actions_mean)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        self._debug("Evaluate", critic_observations=critic_observations, value=value)
        return value

    def _debug(self, message, **values):
        if not self.debug_enabled:
            return
        details = [f"[DreamWaQ:ActorCritic] {message}"]
        for name, value in values.items():
            details.append(tensor_summary(name, value))
        print(" | ".join(details))

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None