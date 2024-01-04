import random

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from models import VQModel, AutoencoderKL
from models.ldm import LatentDiffusion
from models.modules.ddim import DDIMSampler
from trainers.t_base import BaseTrainer
from utils.model import get_ae, load_ae_checkpoint, load_weights


class LDMTrainer(BaseTrainer):
    def __init__(self, lr, default_kwargs, ldm_kwargs, ae_kwargs, resume=None):
        super().__init__(**default_kwargs)

        assert (
            ae_kwargs["ae_ckpt"] is not None
        ), "Specify a checkpoint for the Autoencoder to train LDM"

        # setup autoencoder
        self.ae = get_ae(ae_kwargs, ldm_kwargs["logvar_init"]).to(self.device)
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False

        load_ae_checkpoint(self.ae, ae_kwargs["ae_ckpt"], self.device)

        self.ldm = LatentDiffusion(**ldm_kwargs).to(self.device)
        self.num_timesteps = self.ldm.num_timesteps
        self.opt = AdamW(self.ldm.parameters(), lr)

        self.init_checkpoint(resume)
        self.l_simple_weight = ldm_kwargs["l_simple_weight"]
        self.original_elbo_weight = ldm_kwargs["original_elbo_weight"]
        self.logvar = torch.full(
            fill_value=ldm_kwargs["logvar_init"],
            size=(self.num_timesteps,),
            device=self.device,
        )
        self.epoch_losses = ["loss", "loss_simple", "loss_vlb"]

        self.ddim_steps = ldm_kwargs["ddim_steps"]
        self.ddim_eta = ldm_kwargs["ddim_eta"]

        self.is_unconditional = ldm_kwargs["unet_config"]["is_unconditional"]
        self.use_cross_attn = ldm_kwargs["unet_config"]["use_cross_attn"]
        self.gen_masks = (
            ldm_kwargs["unet_config"]["gen_masks"]
            if "gen_masks" in ldm_kwargs["unet_config"]
            else False
        )

    def _make_ckpt(self, epoch, loss):
        return {
            "epoch": epoch,
            "loss": loss,
            "ldm": self.ldm.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        load_weights(self.ldm, checkpoint["ldm"], strict=False)
        self.start_epoch = checkpoint["epoch"]
        if "opt" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt"])

    @torch.no_grad()
    def encode_to_z(self, x):
        z = self.ae.encode(x)
        if isinstance(self.ae, AutoencoderKL):
            z = z.sample()
        if isinstance(self.ae, VQModel):
            z = z[0]
        return z

    @torch.no_grad()
    def decode_to_x(self, z):
        return self.ae.decode(z)

    def encode_cond(self, mask, image):
        if self.is_unconditional:
            return {"global": None, "local": None}
        cond = {}
        # zmask = self.ldm.mask_encoder(mask)
        zmask = self.encode_to_z(mask)
        cond["local"] = zmask
        drop_global = random.random() > 0.5

        if self.use_cross_attn:
            cond["global"] = self.ldm.img_encoder(image)
            if drop_global:
                cond["global"] = torch.zeros_like(cond["global"])
        else:
            cond["global"] = None
        return cond

    def train_one_epoch(self):
        epoch_loss_dict = {k: 0 for k in self.epoch_losses}
        self.ldm.train()
        for batch in tqdm(self.train_loader):
            loss, loss_dict = self.run_step(batch)
            epoch_loss_dict = {
                k: v + loss_dict[k].item() for k, v in epoch_loss_dict.items()
            }

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.debug:
                break
        # Aggregating and logging metrics
        epoch_loss_dict = {
            k: v / len(self.train_loader) for k, v in epoch_loss_dict.items()
        }
        return epoch_loss_dict

    def eval_one_epoch(self):
        epoch_loss_dict = {k: 0 for k in self.epoch_losses}
        self.ldm.eval()
        samples = {}
        bs = self.test_loader.batch_size
        shape = [
            self.ldm.unet.out_channels,
            self.ldm.unet.image_size,
            self.ldm.unet.image_size,
        ]

        batch = next(iter(self.test_loader))
        gt, mask = batch
        gt = gt.to(self.device)
        mask = mask.to(self.device)
        # mask = mask_to_rgb(mask) TODO Restore

        if self.gen_masks:
            gt = mask

        ddim = DDIMSampler(self.ldm)

        with torch.no_grad():
            _, loss_dict = self.run_step(batch)
        epoch_loss_dict = {k: v + loss_dict[k] for k, v in epoch_loss_dict.items()}
        c = self.encode_cond(mask, gt)
        gen_latents, _ = ddim.sample(self.ddim_steps, bs, shape, c, eta=self.ddim_eta)

        generated = self.decode_to_x(gen_latents)
        samples = {"gt": gt, "gen": generated, "cond": mask}

        # Aggregating and logging metrics
        epoch_loss_dict = {k: v for k, v in epoch_loss_dict.items()}
        return epoch_loss_dict, samples

    def run_step(self, batch):
        x, mask = batch
        x = x.to(self.device)
        mask = mask.to(self.device)
        # mask = mask_to_rgb(mask) TODO Restore

        if self.gen_masks:
            x = mask

        z = self.encode_to_z(x)
        c = self.encode_cond(mask, x)
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()

        out, noise = self.ldm(z, c=c, t=t)

        target = noise

        loss_dict = {}

        loss_simple = F.mse_loss(out, target, reduction="none").mean([1, 2, 3])
        loss_dict.update({f"loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = F.mse_loss(out, target, reduction="none").mean(dim=(1, 2, 3))
        loss_vlb = (self.ldm.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"loss": loss})

        return loss, loss_dict
