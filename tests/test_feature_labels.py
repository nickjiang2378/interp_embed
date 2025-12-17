#!/usr/bin/env python3
from interp_embed.sae.local_sae import GoodfireSAE


def test_goodfire_sae_labels():
    goodfire_sae = GoodfireSAE(
        variant_name="Llama-3.1-8B-Instruct-SAE-l19",
    )
    goodfire_sae.load_feature_labels()
    assert len(goodfire_sae.feature_labels()) > 0
