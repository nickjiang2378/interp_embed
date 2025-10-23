from .base_sae import SAEType

def load_sae_from_metadata(metadata):
  if "sae_type" not in metadata:
    raise Exception(f"Field 'sae_type' not found in metadata.")

  sae_metadata = {k:v for k, v in metadata.items() if k != "sae_type"}
  if metadata["sae_type"] == SAEType.LOCAL:
    from .local_sae import LocalSAE
    return LocalSAE.from_metadata(sae_metadata)
  elif metadata["sae_type"] == SAEType.GOODFIRE_API:
    from .api_sae import GoodfireApiSAE
    return GoodfireApiSAE.from_metadata(sae_metadata)
  elif metadata["sae_type"] == SAEType.GOODFIRE:
    from .local_sae import GoodfireSAE
    return GoodfireSAE.from_metadata(sae_metadata)
  else:
    raise Exception(f"Unrecognized SAE type: {metadata['sae_type']}")