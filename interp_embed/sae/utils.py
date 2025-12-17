import json
from huggingface_hub import hf_hub_download
import re
import torch


def process_device_config(device):
    if isinstance(device, str):
        assert (
            device != "auto"
        ), f"`auto` is not a valid device for the SAE. Please reformat your device config as {{ 'model': 'auto', 'sae': <sae_device> }}"
        return device, device
    elif isinstance(device, dict):
        assert set(device.keys()) == {
            "model",
            "sae",
        }, f"Only 'model' and 'sae' are allowed as device keys: {list(device.keys())}"
        assert isinstance(device["model"], str), f"Model device must be a string"
        assert isinstance(device["sae"], str), f"SAE device must be a string"
        assert device["sae"] != "auto", f"`auto` is not a valid device for the SAE."
        return device["model"], device["sae"]
    else:
        raise TypeError(f"Unrecognized device type, {type(device)}")


def ensure_loaded(func):
    def wrapper(self, *args, **kwargs):
        assert self.is_loaded(), "Load the SAE before using this method!"
        result = func(self, *args, **kwargs)
        return result

    return wrapper


def try_to_load_feature_labels(loc):
    try:
        file_path = hf_hub_download(
            repo_id="nickjiang/feature_labels",
            filename=loc,
        )

        with open(file_path, "r") as f:
            feature_labels = json.load(f)
        return feature_labels
    except Exception as e:
        return dict()


def get_goodfire_d_sae(model_name):
    if "meta-llama/Llama-3.1-8B-Instruct":
        return 4096 * 16
    elif "meta-llama/Llama-3.1-70B-Instruct":
        return 4096 * 8


def get_goodfire_config_from_hf(
    repo_id: str,  # noqa: ARG001
    folder_name: str,
    device: str,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides=None,
    use_8b_model=True,
):
    """Get config for Goodfire SAEs."""

    match = re.search(r"l(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not find layer number in filename: {folder_name}")
    layer = int(match.group(1))

    model_name = (
        "meta-llama/Llama-3.1-8B-Instruct"
        if use_8b_model
        else "meta-llama/Llama-3.3-70B-Instruct"
    )

    return {
        "architecture": "standard",
        "d_in": 4096,  # LLaMA 8B hidden size
        "d_sae": get_goodfire_d_sae(
            model_name
        ),  # Expansion factor 16 for 8B model, 8 for 70B model
        "dtype": "float32",
        "context_size": 128000,
        "model_name": model_name,
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": None,
        "activation_fn": "relu",
        "normalize_activations": "none",
        "device": device,
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
        **(cfg_overrides or {}),
    }


def goodfire_sae_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides=None,
):
    """Load a Goodfire SAE."""
    if repo_id == "Goodfire/Llama-3.1-8B-Instruct-SAE-l19":
        use_8b_model = True
    elif repo_id == "Goodfire/Llama-3.3-70B-Instruct-SAE-l50":
        use_8b_model = False
    else:
        raise ValueError(f"Invalid repo_id for Goodfire SAE: {repo_id}")

    # Download weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=folder_name,
        force_download=force_download,
    )

    # Load state dict
    state_dict_loaded = torch.load(sae_path, map_location=device)

    # Create config
    cfg_dict = get_goodfire_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
        use_8b_model,
    )

    # Convert weights
    state_dict = {
        "W_enc": state_dict_loaded["encoder_linear.weight"].T,
        "W_dec": state_dict_loaded["decoder_linear.weight"].T,
        "b_enc": state_dict_loaded["encoder_linear.bias"],
        "b_dec": state_dict_loaded["decoder_linear.bias"],
    }

    return cfg_dict, state_dict, None


def store_activations_hook(model, input, output, activations, name):
    # Store the output activation
    activations[name] = (
        output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
    )
