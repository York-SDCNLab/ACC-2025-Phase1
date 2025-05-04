import torch
import torch.nn as nn

class RepresentationModel(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        min_std=0.1,
        max_std=1.00
    ):
        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.min_std = min_std
        self.max_std = max_std

        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(True),
            nn.Linear(in_dim, 2*latent_dim)
        )

    def forward(self, x):
        y = self.model(x)
        mu, log_sigma = torch.split(y, self.latent_dim, dim=-1)
        sigma = (self.max_std - self.min_std) * torch.sigmoid(log_sigma / (self.max_std - self.min_std)) + self.min_std

        #sample from dist
        noise = torch.randn_like(mu)
        sample = mu + noise*sigma

        return sample, mu, sigma

class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim,
        action_dim,
        hidden_dim,
        latent_dim,
        action_latent_dim
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_latent_dim = action_latent_dim

        self.pre_gru_layer = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(True)
        )

        self.recurrent_model = nn.GRUCell(
            input_size = hidden_dim,
            hidden_size = hidden_dim
        )

        self.prior_action_layer = nn.Sequential(
            nn.Linear(action_dim, action_latent_dim),
            nn.LeakyReLU(True)
        )

        self.posterior_action_layer = nn.Sequential(
            nn.Linear(action_dim, action_latent_dim),
            nn.LeakyReLU(True)
        )

        self.prior = RepresentationModel(
            in_dim = hidden_dim + action_latent_dim,
            latent_dim = latent_dim
        )

        self.posterior = RepresentationModel(
            in_dim = hidden_dim + embed_dim + action_latent_dim,
            latent_dim = latent_dim
        )

    def dream(
        self,
        action,
        in_state
    ):
        h, z = in_state

        #get next hidden state
        in_t = self.pre_gru_layer(z)
        h = self.recurrent_model(in_t, h)

        #compute prior
        prior_latent_action_t = self.prior_action_layer(action)
        prior_sample_t, prior_mu_t, prior_sigma_t = self.prior(torch.cat([h, prior_latent_action_t], dim=-1))

        return h, prior_mu_t, prior_sigma_t, prior_sample_t

    def forward(
        self,
        embeds,
        actions,
        in_state
    ):
        T, B = embeds.shape[:2]
        (h, z) = in_state

        hidden_states = []
        prior_mus = []
        prior_sigmas = []
        prior_samples = []
        posterior_mus = []
        posterior_sigmas = []
        posterior_samples = []
        for t in range(T):
            #get next hidden state
            in_t = self.pre_gru_layer(z)
            h = self.recurrent_model(in_t, h)

            #compute prior
            prior_latent_action_t = self.prior_action_layer(actions[t])
            prior_sample_t, prior_mu_t, prior_sigma_t = self.prior(torch.cat([h, prior_latent_action_t], dim=-1))

            #compute posterior
            posterior_latent_action_t = self.posterior_action_layer(actions[t])
            posterior_sample_t, posterior_mu_t, posterior_sigma_t = self.posterior(torch.cat([h, embeds[t], posterior_latent_action_t], dim=-1))

            #record tensors
            hidden_states.append(h)
            prior_mus.append(prior_mu_t)
            prior_sigmas.append(prior_sigma_t)
            prior_samples.append(prior_sample_t)
            posterior_mus.append(posterior_mu_t)
            posterior_sigmas.append(posterior_sigma_t)
            posterior_samples.append(posterior_sample_t)

        hidden_states = torch.stack(hidden_states)
        prior_mus = torch.stack(prior_mus)
        prior_sigmas = torch.stack(prior_sigmas)
        prior_samples = torch.stack(prior_samples)
        posterior_mus = torch.stack(posterior_mus)
        posterior_sigmas = torch.stack(posterior_sigmas)
        posterior_samples = torch.stack(posterior_samples)

        return hidden_states, \
                prior_mus, \
                prior_sigmas, \
                prior_samples, \
                posterior_mus, \
                posterior_sigmas, \
                posterior_samples