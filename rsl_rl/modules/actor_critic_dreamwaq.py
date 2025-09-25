import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCriticDreamWaQ(nn.Module):
    """Actor-critic module with a single encoder and multi-head beta-VAE decoder."""
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
                        decoder_head_dims=None,
                        decoder_head_hidden_dims=None,
                        **kwargs):
        if kwargs:
            print("ActorCriticDreamWaQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDreamWaQ, self).__init__()

        self.activation_name = activation

        # CENet
        cenet_input_dim = num_actor_obs * (history_len + 1)

        # Encoder
        self.cenet_encoder = nn.Sequential(
            nn.Linear(cenet_input_dim, 128),
            self._activation(),
            nn.Linear(128, 64),
            self._activation()
        )

        # Latent head (mu, logvar)
        self.latent_mu_head = nn.Linear(64, latent_dims)
        self.latent_logvar_head = nn.Linear(64, latent_dims)

        # Decoder heads (multi-head beta-VAE)
        if decoder_head_dims is None:
            decoder_head_dims = {
                "reconstruction": num_actor_obs,
                "velocity": velocity_dims,
            }

        default_decoder_hidden = {
            "reconstruction": [64, 128],
            "velocity": [64],
            "default": [64, 128],
        }

        if decoder_head_hidden_dims is None:
            decoder_head_hidden_dims = {}
        elif isinstance(decoder_head_hidden_dims, (list, tuple)):
            decoder_head_hidden_dims = {name: list(decoder_head_hidden_dims) for name in decoder_head_dims.keys()}

        def _resolve_hidden_dims(head_name):
            if head_name in decoder_head_hidden_dims:
                dims = decoder_head_hidden_dims[head_name]
            elif "default" in decoder_head_hidden_dims:
                dims = decoder_head_hidden_dims["default"]
            else:
                dims = default_decoder_hidden.get(head_name, default_decoder_hidden["default"])
            return list(dims)

        self.decoder_head_dims = decoder_head_dims
        self.decoder_heads = nn.ModuleDict()
        for head_name, head_dim in decoder_head_dims.items():
            hidden_dims = _resolve_hidden_dims(head_name)
            self.decoder_heads[head_name] = self._build_mlp(latent_dims, hidden_dims, head_dim)

        print(f"[DreamWaQ] CENet Encoder: {self.cenet_encoder}")
        print(f"[DreamWaQ] CENet Decoder Heads: {list(self.decoder_heads.keys())}")

        mlp_input_dim_a = num_actor_obs + velocity_dims + latent_dims
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(self._activation())
        for l in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_layers.append(self._activation())
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(self._activation())
        for l in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
            critic_layers.append(self._activation())
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        print(f"[DreamWaQ] Actor MLP: {self.actor}")
        print(f"[DreamWaQ] Critic MLP: {self.critic}")

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


    def _activation(self):
        activation = get_activation(self.activation_name)
        if activation is None:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        return activation

    def _build_mlp(self, input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)


    def reset(self, dones=None):
        pass

    def forward(self, history_observations):
        batch_size = history_observations.shape[0]
        history_observations_flat = history_observations.view(batch_size, -1)
        encoded = self.cenet_encoder(history_observations_flat)

        latent_mu = self.latent_mu_head(encoded)
        latent_logvar = torch.clamp(self.latent_logvar_head(encoded), -10, 10)

        latent_std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(latent_std)
        z = latent_mu + eps * latent_std

        decoder_outputs = {}
        for head_name, head in self.decoder_heads.items():
            decoder_outputs[head_name] = head(z)

        predicted_velocity = decoder_outputs.get("velocity", None)
        if predicted_velocity is None:
            raise KeyError("Decoder outputs must include a 'velocity' head to compute actions.")

        return predicted_velocity, z, decoder_outputs, latent_mu, latent_logvar

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

    def act(self, observations, history_observations, **kwargs):

        predicted_velocity, z, _, _, _ = self.forward(history_observations)
        actor_input = torch.cat([observations, predicted_velocity, z], dim=-1)

        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history_observations):
        predicted_velocity, z, _, _, _ = self.forward(history_observations)
        actor_input = torch.cat([observations, predicted_velocity, z], dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

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
