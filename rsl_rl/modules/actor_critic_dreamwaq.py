import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCriticDreamWaQ(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        history_len,
                        velocity_dim=3,
                        latent_dim=16,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticDreamWaQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDreamWaQ, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs + velocity_dim + latent_dim
        mlp_input_dim_c = num_critic_obs

        self.cenet = CENet(num_actor_obs, history_len)

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

        print(f"[Asymmetric] Actor MLP: {self.actor}")
        print(f"[Asymmetric] Critic MLP: {self.critic}")

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
        pass

    def forward(self):
        raise NotImplementedError
    
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, velocity, latent):
        mean = self.actor(torch.cat([observations, velocity, latent], dim=-1))
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, velocity, latent, **kwargs):
        self.update_distribution(observations, velocity, latent)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, velocity, latent):
        actions_mean = self.actor(observations, velocity, latent)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
class CENet(torch.nn.Module):
    def __init__(self,  num_actor_obs,
                        history_len,
                        velocity_dim=3,
                        latent_dim=16,
                        num_heads=2, # velocity + latent
                        encoder_hidden_dims=[128, 64],
                        decoder_hidden_dims=[64, 128],
                        beta=0.4, # Need to be revised
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("CENet.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(CENet, self).__init__()
        
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.beta = beta

        activation = get_activation(activation)

        mlp_input_dim_e = num_actor_obs * history_len
        mlp_input_dim_a = num_actor_obs

        # CENet - Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_e, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], velocity_dim + latent_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers) # Context vector

        # For VAE
        self.fc_mu = nn.Linear(velocity_dim + latent_dim, velocity_dim + latent_dim)
        self.fc_logvar = nn.Linear(velocity_dim + latent_dim, velocity_dim + latent_dim)

        # CENet - Multihead Decoder
        # Head 1: velocity
        velocity_layers = []
        velocity_layers.append(nn.Linear(velocity_dim, decoder_hidden_dims[0]))
        velocity_layers.append(activation)
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                velocity_layers.append(nn.Linear(decoder_hidden_dims[l], mlp_input_dim_a))
            else:
                velocity_layers.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                velocity_layers.append(activation)
        self.velocity_head = nn.Sequential(*velocity_layers)

        # Head 2: latent
        latent_layers = []
        latent_layers.append(nn.Linear(latent_dim, decoder_hidden_dims[0]))
        latent_layers.append(activation)
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                latent_layers.append(nn.Linear(decoder_hidden_dims[l], mlp_input_dim_a))
            else:
                latent_layers.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                latent_layers.append(activation)
        self.latent_head = nn.Sequential(*latent_layers)

        self.decoder = nn.Linear(num_heads * mlp_input_dim_a, mlp_input_dim_a)

    def forward(self, history_obs):
        h = self.encoder(history_obs)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std # reparameterization

        v_enc, z_enc = torch.split(z, [self.velocity_dim, self.latent_dim], dim=-1)
        v_out = self.velocity_head(v_enc)
        z_out = self.latent_head(z_enc)
        obs_est = self.decoder(torch.cat((v_out, z_out), dim=-1))

        return v_enc, z_enc, obs_est, mu, logvar
    
    def encode(self, history_obs):
        z = self.encoder(history_obs)
        v_enc, z_enc = torch.split(z, [self.velocity_dim, self.latent_dim], dim=-1)
        return v_enc, z_enc

    def compute_loss(self, v_est, v_truth, obs_est, obs_truth, mu, logvar):
        # body velocity estimation loss
        est_loss = F.mse_loss(v_est, v_truth, reduction='mean')

        # VAE loss
        recon_loss = F.mse_loss(obs_est, obs_truth, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + self.beta * kl_loss

        return est_loss + vae_loss

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