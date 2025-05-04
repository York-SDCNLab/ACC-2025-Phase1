import torch
import numpy as np
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from typing import Any, Tuple, Dict

from .rssm import RSSM
from .encoders import MultiEncoder
from .decoders import DenseCategoricalDecoder

class WorldModel(nn.Module):
    def __init__(
        self,
        deter_dim = 256,
        device="cuda:0"
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.device = device

        self.encoder = MultiEncoder(
            include_image=True,
            include_vector=True
        )

        self.rssm = RSSM(
            embed_dim = self.encoder.out_dim,
            action_dim = 2,
            hidden_dim = 128,
            latent_dim = deter_dim,
            action_latent_dim = 64
        )

        '''self.waypoint_decoder = DenseDecoder(
            in_dim = self.rssm.latent_dim + self.rssm.hidden_dim, 
            out_dim = 200,
            hidden_dim = deter_dim,
            hidden_layers = 2
        )'''

        self.fsm_decoder = DenseCategoricalDecoder(
            in_dim = self.rssm.latent_dim + self.rssm.hidden_dim,
            num_channels = 1,
            num_classes = 4,
            hidden_dim = deter_dim,
            hidden_layers = 2
            #obstcal=False
        )

        #set nominal weight low since it occurs much more frequently in data
        self.fsm_decoder.set_criterion_weights(torch.tensor([0.1, 1.0, 1.0, 1.0], device=self.device))

        '''self.state_decoder = DenseDecoder(
            in_dim = self.rssm.latent_dim + self.rssm.hidden_dim,
            out_dim = 5,
            hidden_dim = deter_dim,
            hidden_layers = 2
        )'''

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        optimizer_encoder = torch.optim.AdamW(self.encoder.parameters(), lr=lr, eps=eps)
        #optimizer_image_decoder = torch.optim.AdamW(self.image_decoder.parameters(), lr=lr, eps=eps)
        #optimizer_waypoint_decoder = torch.optim.AdamW(self.waypoint_decoder.parameters(), lr=lr, eps=eps)
        optimizer_fsm_decoder = torch.optim.AdamW(self.fsm_decoder.parameters(), lr=lr, eps=eps)
        #optimizer_state_decoder = torch.optim.AdamW(self.state_decoder.parameters(), lr=lr, eps=eps)
        optimizer_rssm = torch.optim.AdamW(self.rssm.parameters(), lr=lr, eps=eps)
        return optimizer_encoder, optimizer_fsm_decoder, optimizer_rssm
    
    def grad_clip(self, grad_clip, grad_clip_ac=None):
        grad_metrics = {
            'grad_norm_encoder': nn.utils.clip_grad_norm_(self.encoder.parameters(), grad_clip),
            #'grad_norm_waypoint_decoder': nn.utils.clip_grad_norm_(self.waypoint_decoder.parameters(), grad_clip),
            'grad_norm_fsm_decoder': nn.utils.clip_grad_norm_(self.fsm_decoder.parameters(), grad_clip),
            #'grad_norm_state_decoder': nn.utils.clip_grad_norm_(self.state_decoder.parameters(), grad_clip),
            'grad_norm_rssm': nn.utils.clip_grad_norm_(self.rssm.parameters(), grad_clip)
        }
        return grad_metrics
    
    def init_state(self, batch_length: int = 1, batch_size: int = 1):
        return (
            torch.zeros((batch_size, self.rssm.hidden_dim), device=self.device),
            torch.zeros((batch_size, self.deter_dim), device=self.device)
        )
    
    #for inference
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        in_state: Any
    ):
        metrics = {}
        pred = {}

        #add batch and time dims
        for k in obs:
            if isinstance(obs[k], np.ndarray):
                if k == "image":
                    obs[k] = obs[k].astype(np.float32)
                    obs[k] = obs[k] / 255.0 - 0.5
                    obs[k] = obs[k].transpose(2, 0, 1) #(H, W, C) => (C, H, W)
                
                obs[k] = torch.from_numpy(obs[k][None, None]).to(self.device)

        embed = self.encoder(obs)
        hidden_states, prior_mus, prior_sigmas, prior_samples, posterior_mus, posterior_sigmas, posterior_samples = self.rssm(
            embed,
            obs["action"],
            in_state
        )
        out_state = (hidden_states[-1], posterior_mus[-1])
        features = torch.cat([hidden_states, posterior_mus], dim=-1)

        #image = self.image_decoder(features)
        #state_pred = self.state_decoder(features)
        fsm_pred = self.fsm_decoder(features)
        #waypoint_pred = self.waypoint_decoder(features)

        #state_yaw = torch.arctan2(state_pred[..., [3]], state_pred[..., [2]])
        #state = torch.cat([state_pred[..., :2], state_yaw, state_pred[..., [4]]], dim=-1)
        
        with torch.no_grad():
            #pred["image"] = image.detach()
            #pred["state"] = state.detach()
            pred["fsm"] = fsm_pred.softmax(dim=-1).detach()
            #pred["waypoints"] = waypoint_pred.detach().reshape(199, 2)

        return pred, out_state, metrics
    
    def training_step(
        self,
        obs: Dict[str, torch.Tensor],
        in_state: Any
    ):
        metrics = {}
        pred = {}

        embed = self.encoder(obs)
        hidden_states, prior_mus, prior_sigmas, prior_samples, posterior_mus, posterior_sigmas, posterior_samples = self.rssm(
            embed,
            obs["action"],
            in_state
        )

        #get features
        features = torch.cat([hidden_states, posterior_mus], dim=-1)
        #state_pred = self.state_decoder(features)
        fsm_pred = self.fsm_decoder(features)

        #compute state loss
        '''target_cos = torch.cos(obs["state"][..., [2]])
        target_sin = torch.sin(obs["state"][..., [2]])
        state_target = torch.cat([obs["state"][..., :2], target_cos, target_sin, obs["state"][..., [3]]], dim=-1)

        loss_state = self.state_decoder.loss(state_pred, state_target)'''

        n_step, n_batch = obs["fsm"].shape
        loss_fsm = self.fsm_decoder.loss(fsm_pred.squeeze(1), obs["fsm"].to(torch.long).reshape(n_step*n_batch))      

        #posterior loss
        posterior_var = posterior_sigmas[1:] ** 2
        prior_var = prior_sigmas.detach()[1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigmas[1:])
        prior_log_sigma = torch.log(prior_sigmas.detach()[1:])

        kl_div = (
            prior_log_sigma - posterior_log_sigma - 0.5
            + (posterior_var + (posterior_mus[1:] - prior_mus.detach()[1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:1] - 0.5 + (posterior_var[:1] + posterior_mus[:1] ** 2) / 2
        kl_div_posterior = torch.cat([first_kl, kl_div], dim=0)

        #prior loss
        posterior_var = posterior_sigmas.detach()[1:] ** 2
        prior_var = prior_sigmas[1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigmas.detach()[1:])
        prior_log_sigma = torch.log(prior_sigmas[1:])

        kl_div = (
            prior_log_sigma - posterior_log_sigma - 0.5
            + (posterior_var + (posterior_mus.detach()[1:] - prior_mus[1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:1] - 0.5 + (posterior_var[:1] + posterior_mus.detach()[:1] ** 2) / 2
        kl_div_prior = torch.cat([first_kl, kl_div], dim=0)
        kl_loss = (0.8*kl_div_prior) + (0.2*kl_div_posterior)

        #aggregate losses
        loss = 0.0
        loss += kl_loss.mean()
        #loss += loss_state.mean() 
        loss += loss_fsm.mean()

        with torch.no_grad():
            metrics.update({
                #"loss_image": loss_image.mean().detach(),
                "loss_fsm": loss_fsm.mean().detach(),
                #"loss_state": loss_state.mean().detach(),
                "loss_transition": kl_loss.mean().detach(),
                #"loss_waypoints": loss_waypoints.sum().detach(),
                "loss": loss.detach()
            })

        return loss, metrics
    
    def training_step_autoregressive(
        self,
        obs: Dict[str, torch.Tensor],
        in_state: Any
    ):
        metrics = {}
        pred = {}

        hidden_states = []
        prior_mus = []
        prior_sigmas = []
        prior_samples = []
        posterior_mus = []
        posterior_sigmas = []
        posterior_samples = []

        fsm_preds = []
        state_preds = []
        waypoint_preds = []

        h, z = in_state
        n_step = obs["image"].shape[0]
        for t in range(n_step):
            if t == 0:
                obs_t = {
                    "image": obs["image"][t].unsqueeze(0),
                    "image_valid": obs["image_valid"][t].unsqueeze(0),
                    "state": F.pad(obs["state"][t].unsqueeze(0), (0, 1), value=1.0), #valid gps on start
                    "waypoints": obs["global_waypoints"][t].unsqueeze(0),
                    "gps_valid": obs["gps_valid"][t].unsqueeze(0),
                    "hardware_metrics": obs["hardware_metrics"][t].unsqueeze(0),
                }
            else:
                gps_valid = obs["gps_valid"][t].unsqueeze(0)
                state_yaw = torch.arctan2(state_pred[..., [3]], state_pred[..., [2]])
                state = torch.cat([state_pred[..., :2], state_yaw, state_pred[..., [4]], gps_valid.unsqueeze(-1)], dim=-1) 
                obs_t = {
                    "image": obs["image"][t].unsqueeze(0),
                    "image_valid": obs["image_valid"][t].unsqueeze(0),
                    "state": state, #autoregressive state update
                    "waypoints": obs["global_waypoints"][t].unsqueeze(0),
                    "gps_valid": gps_valid,
                    "hardware_metrics": obs["hardware_metrics"][t].unsqueeze(0),
                }

            embed_t = self.encoder(obs_t)

            #get next hidden state
            in_t = self.rssm.pre_gru_layer(z)
            h = self.rssm.recurrent_model(in_t, h)

            #compute prior
            prior_latent_action_t = self.rssm.prior_action_layer(obs["action"][t])
            prior_sample_t, prior_mu_t, prior_sigma_t = self.rssm.prior(torch.cat([h, prior_latent_action_t], dim=-1))

            #compute posterior
            posterior_latent_action_t = self.rssm.posterior_action_layer(obs["action"][t])
            posterior_sample_t, posterior_mu_t, posterior_sigma_t = self.rssm.posterior(torch.cat([h, embed_t.squeeze(0), posterior_latent_action_t], dim=-1))

            feature = torch.cat([h, posterior_mu_t], dim=-1)
            fsm_pred = self.fsm_decoder(feature.unsqueeze(0))
            state_pred = self.state_decoder(feature.unsqueeze(0))
            #waypoint_pred = self.waypoint_decoder(feature.unsqueeze(0))
            fsm_preds.append(fsm_pred)
            state_preds.append(state_pred)
            #waypoint_preds.append(waypoint_pred)

            #record tensors
            hidden_states.append(h)
            prior_mus.append(prior_mu_t)
            prior_sigmas.append(prior_sigma_t)
            prior_samples.append(prior_sample_t)
            posterior_mus.append(posterior_mu_t)
            posterior_sigmas.append(posterior_sigma_t)
            posterior_samples.append(posterior_sample_t)

        fsm_preds = torch.stack(fsm_preds)
        state_preds = torch.stack(state_preds)
        #waypoint_preds = torch.stack(waypoint_preds)

        hidden_states = torch.stack(hidden_states)
        prior_mus = torch.stack(prior_mus)
        prior_sigmas = torch.stack(prior_sigmas)
        prior_samples = torch.stack(prior_samples)
        posterior_mus = torch.stack(posterior_mus)
        posterior_sigmas = torch.stack(posterior_sigmas)
        posterior_samples = torch.stack(posterior_samples)

        #compute state loss
        target_cos = torch.cos(obs["gt_state"][..., [2]])
        target_sin = torch.sin(obs["gt_state"][..., [2]])
        state_target = torch.cat([obs["gt_state"][..., :2], target_cos, target_sin, obs["gt_state"][..., [3]]], dim=-1)

        loss_state = self.state_decoder.loss(state_preds.squeeze(1), state_target)

        n_step, n_batch = obs["fsm"].shape
        loss_fsm = self.fsm_decoder.loss(fsm_preds.squeeze(2).reshape(n_step*n_batch, -1), obs["fsm"].to(torch.long).reshape(n_step*n_batch))      

        '''waypoints = obs["global_waypoints"].reshape(n_step*n_batch, 100, 2)
        loss_waypoints = 100 * self.waypoint_decoder.loss(
            waypoint_preds.reshape(-1, self.waypoint_decoder.out_dim // 2, 2), 
            waypoints
        )'''

        #posterior loss
        posterior_var = posterior_sigmas[1:] ** 2
        prior_var = prior_sigmas.detach()[1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigmas[1:])
        prior_log_sigma = torch.log(prior_sigmas.detach()[1:])

        kl_div = (
            prior_log_sigma - posterior_log_sigma - 0.5
            + (posterior_var + (posterior_mus[1:] - prior_mus.detach()[1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:1] - 0.5 + (posterior_var[:1] + posterior_mus[:1] ** 2) / 2
        kl_div_posterior = torch.cat([first_kl, kl_div], dim=0)

        #prior loss
        posterior_var = posterior_sigmas.detach()[1:] ** 2
        prior_var = prior_sigmas[1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigmas.detach()[1:])
        prior_log_sigma = torch.log(prior_sigmas[1:])

        kl_div = (
            prior_log_sigma - posterior_log_sigma - 0.5
            + (posterior_var + (posterior_mus.detach()[1:] - prior_mus[1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:1] - 0.5 + (posterior_var[:1] + posterior_mus.detach()[:1] ** 2) / 2
        kl_div_prior = torch.cat([first_kl, kl_div], dim=0)
        kl_loss = (0.8*kl_div_prior) + (0.2*kl_div_posterior)

        #aggregate losses
        loss = 0.0
        loss += kl_loss.mean()
        loss += loss_state.mean() 
        loss += loss_fsm.mean()
        #loss += loss_waypoints.mean()

        with torch.no_grad():
            metrics.update({
                #"loss_image": loss_image.mean().detach(),
                "loss_fsm": loss_fsm.mean().detach(),
                "loss_state": loss_state.mean().detach(),
                "loss_transition": kl_loss.mean().detach(),
                #"loss_waypoints": loss_waypoints.sum().detach(),
                "loss": loss.detach()
            })

        return loss, metrics