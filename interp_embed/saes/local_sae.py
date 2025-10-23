import numpy as np
from sae_lens import SAE as SAEModel
from transformer_lens import HookedTransformer
import torch
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from functools import partial
import warnings

from .base_sae import BaseSAE, SAEType
from .utils import process_device_config, ensure_loaded, try_to_load_feature_labels, goodfire_sae_loader, store_activations_hook

class LocalSAE(BaseSAE):
  def __init__(self, sae_id = "blocks.8.hook_resid_pre", release = "gpt2-small-res-jb", device = "cuda:0", **kwargs):
    super().__init__(**kwargs)
    self.model_device, self.sae_device = process_device_config(device)
    self.sae_id = sae_id
    self.release = release
    self.model = None
    self.sae = None
    self.tokenizer = None

  def metadata(self):
    parent_metadata = super().metadata()
    parent_metadata.update({
      "sae_id": self.sae_id,
      "release": self.release,
      "device": {
        "model": self.model_device,
        "sae": self.sae_device
      },
      "sae_type": SAEType.LOCAL
    })
    return parent_metadata

  def load_models(self):
    print("Loading SAE...")
    self.sae = SAEModel.from_pretrained(
      release=self.release,  # see other options in sae_lens/pretrained_saes.yaml
      sae_id=self.sae_id,  # won't always be a hook point
      device=self.sae_device,
    )
    print("Loading language model...")
    self.model = HookedTransformer.from_pretrained(self.sae.cfg.metadata.model_name, device=self.model_device)
    self.tokenizer = self.model.tokenizer

  @ensure_loaded
  @torch.no_grad()
  def encode(self, texts):
    assert len(texts) > 0, "There must be more than one text to encode."
    self.sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    self.tokenizer.pad_token = self.tokenizer.eos_token
    tokens = self.tokenizer(
      texts,
      padding="longest",   # pad to the longest sequence in the batch
      truncation=self.truncate, # TODO: if the number of tokens exceeds the context window, you should filter this out to avoid erring the rest of the batch
      return_tensors="pt"
    )

    _, cache = self.model.run_with_cache(tokens["input_ids"], prepend_bos=True)

    # Use the SAE
    feature_acts = self.sae.encode(cache[self.sae.cfg.metadata.hook_name].to(self.sae_device))

    feature_acts_np = feature_acts.detach().cpu().numpy()
    attn_mask = tokens["attention_mask"].numpy().astype(bool)
    return [csr_matrix(feature_acts_np[i][attn_mask[i]]) for i in range(feature_acts_np.shape[0])]

  @ensure_loaded
  def encode_chat(self, chat_conversations):
    assert self.chat_template_exists(), "Chat template does not exist for this model's tokenizer"
    texts = [self.tokenizer.apply_chat_template(chat_conversation, tokenize=False) for chat_conversation in chat_conversations]
    return self.encode(texts)

  def destroy_models(self):
    self.sae = None
    self.model = None

class GoodfireSAE(BaseSAE):
  def __init__(self, variant_name: str = "Llama-3.1-8B-Instruct-SAE-l19", device = "cuda:0", quantize = False, **kwargs):
    super().__init__(**kwargs)
    self.variant_name = variant_name
    self.quantize = quantize
    self.model_device, self.sae_device = process_device_config(device)
    self.activations = dict()
    self.activation_hook_handle = None

  def metadata(self):
    parent_metadata = super().metadata()
    parent_metadata.update({
      "variant_name": self.variant_name,
      "quantize": self.quantize,
      "device": {
        "model": self.model_device,
        "sae": self.sae_device
      },
      "sae_type": SAEType.GOODFIRE
    })
    return parent_metadata

  def load_models(self):
    # Load the model, sae, and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_8bit = True,
        bnb_8bit_compute_dtype=torch.float32
    )
    # Mapping of variant names to associated config
    variant_configs = {
        # "Llama-3.3-70B-Instruct-SAE-l50": {
        #     "hf_model": "meta-llama/Llama-3.3-70B-Instruct",
        #     "goodfire_release": "Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
        #     "sae_id": "Llama-3.3-70B-Instruct-SAE-l50.pt",
        #     "feature_labels_file": "goodfire/meta-llama/Llama-3.3-70B-Instruct.json",
        #     "device_map": self.model_device
        # },
        "Llama-3.1-8B-Instruct-SAE-l19": {
            "hf_model": "meta-llama/Llama-3.1-8B-Instruct",
            "goodfire_release": "Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
            "sae_id": "Llama-3.1-8B-Instruct-SAE-l19.pth",
            "feature_labels_file": "goodfire/meta-llama/Llama-3.1-8B-Instruct.json",
            "device_map": self.model_device
        }
    }

    if self.variant_name not in variant_configs:
        raise ValueError(f"Variant {self.variant_name} not supported")

    if self.quantize:
      warnings.warn("Quantizing the language model may cause feature activations to be less accurate.")

    config = variant_configs[self.variant_name]

    self.model = AutoModelForCausalLM.from_pretrained(
        config["hf_model"],
        quantization_config=bnb_config if self.quantize else None,
        device_map=config["device_map"]
    )

    # Add hooks to the model
    self.activations = {}
    match = re.search(r"l(\d+)", self.variant_name)
    if match is None:
        raise ValueError(f"Could not find layer number in filename: {self.variant_name}")
    layer = int(match.group(1))
    activation_hook = partial(store_activations_hook, activations=self.activations, name=f"internal")
    self.model.model.layers = torch.nn.ModuleList(self.model.model.layers[:layer+1]) # Truncate the model to the layer we want to extract activations from
    self.activation_hook_handle = self.model.model.layers[layer].register_forward_hook(activation_hook)
    torch.cuda.empty_cache()

    self.tokenizer = AutoTokenizer.from_pretrained(config["hf_model"])
    self.sae = SAEModel.from_pretrained(
        release=config["goodfire_release"],
        sae_id=config["sae_id"],
        device=self.sae_device,
        converter=goodfire_sae_loader,
    )
    self._feature_labels = try_to_load_feature_labels(config["feature_labels_file"])

    # Load the feature labels
    if self._feature_labels:
      self._feature_labels = {int(key): value for key, value in self.feature_labels().items()} # Convert keys to ints

    self.tokenizer.pad_token = self.tokenizer.eos_token

  @ensure_loaded
  def encode(self, texts):
    inputs = self.tokenize(texts, padding=True, as_tokens=False)

    with torch.no_grad():
      outputs = self.model(
        input_ids = torch.tensor(inputs["input_ids"]).to(self.model.device),
        attention_mask = torch.tensor(inputs["attention_mask"]).to(self.model.device)
      )

      feature_acts = self.sae.encode(self.activations["internal"].to(self.sae.device))

    feature_acts_np = feature_acts.float().detach().cpu().numpy()
    attn_mask = np.array(inputs["attention_mask"]).astype(bool)

    # Clean up memory
    del outputs, inputs
    torch.cuda.empty_cache()

    return [csr_matrix(feature_acts_np[i][attn_mask[i]]) for i in range(feature_acts_np.shape[0])]

  def destroy_models(self):
    self.activations = dict()
    self.activation_hook_handle.remove()
    self.model = None
    self.sae = None
