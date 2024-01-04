import yaml
import torch
from models import AutoencoderKL, VQModel

def read_model_config(config_file):
    with open(f"configs/{config_file}", 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            return parsed_yaml["model"]
        except yaml.YAMLError as exc:
            raise exc

def load_weights(model, ckpt, strict=False):
    u, m = model.load_state_dict(ckpt, strict=strict)
    model_name = str(model.__class__).split('.')[-1].split("'")[0]
    print(f"Loading weights for {model_name}")
    if not (u or m):
        print(f"All keys matched")
    if u:
        print(f"Missing keys: {u}")
    if m:
        print(f"Unexpected keys: {m}")
    # return model


def load_ae_checkpoint(ae, ckpt_path, device):
    assert ckpt_path is not None, "A checkpoint of the autoencoder is required to train the LDM"
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'vqmodel' in checkpoint:
        load_weights(ae, checkpoint['vqmodel'], strict=False)
        # ae = load_weights(ae, checkpoint['vqmodel'], strict=False)
        # ae.load_state_dict(
        #     checkpoint['vqmodel'], strict=False)
    elif 'vae' in checkpoint:
        load_weights(ae, checkpoint['vae'], strict=False)
        # ae.load_state_dict(
        #     checkpoint['vae'], strict=False)
    # return ae

def get_ae(ae_kwargs, logvar_init=None):
    ae_type = ae_kwargs["type"]
    if ae_type == "kl":
        ae_kwargs.update({"logvar_init": logvar_init})
        return AutoencoderKL(**ae_kwargs['params'])
    elif ae_type == "vq":
        return VQModel(**ae_kwargs['params'])
    else:
        raise ValueError(f"Unknown Autoencoder type: {ae_type}")