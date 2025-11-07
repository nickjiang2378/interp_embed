# InterpEmbed

`interp_embed` turns unstructured text datasets into interpretable embeddings using sparse autoencoders (SAEs). Each dimension of the embedding corresponds to a human-understandable concept (e.g. a "feature"), such as tone, topic, or reasoning style, making it easy to find granular insights about large corpora. The library lets you compute, store, and analyze these features for tasks like dataset comparison, bias detection, and targeted clustering.

## Setup

After cloning this repository, proceed with either option:

**With uv (recommended):**
```bash
uv sync  # To install uv, see https://docs.astral.sh/uv/getting-started/installation/
```

**Without uv (using pip):**
```bash
pip install -r requirements.txt
```

Create a `.env` file that has `OPENROUTER_API_KEY` and `OPENAI_KEY`. We use these models for creating feature labels if they don't exist.

## Quickstart
First, create a dataset object. We currently support SAEs from SAELens (`LocalSAE`) and Goodfire (`GoodfireSAE`).

```python
from interp_embed import Dataset
from interp_embed.saes.local_sae import GoodfireSAE
import pandas as pd

# 1. Load a Goodfire SAE or SAE supported through the SAELens package
sae = GoodfireSAE(
    variant_name="Llama-3.1-8B-Instruct-SAE-l19",
    device="cuda:0", # optional
)

# 2. Prepare your data as a DataFrame
df = pd.DataFrame({
    "text": ["Good morning!", "Hello there!", "Good afternoon."],
    "date": ["2022-01-10", "2021-08-23", "2023-03-14"]
})

# 3. Create dataset - computes and saves feature activations
dataset = Dataset(
    data=df,
    sae=sae,
    field="text",  # Optional. Column containing text to analyze
    save_path="my_dataset.pkl"  # Optional. Auto-saves progress, which enables recovery if computations fail
)

# 4. In the future, load saved dataset to skip expensive recomputation.
dataset = Dataset.load_from_file("my_dataset.pkl") # # If some activations failed, use 'resume=True' to continue.
```

Here are some commonly used methods.
```python
# Get feature activations as a sparse matrix of shape (N = # documents, F = # features)
embeddings = dataset.latents()

# Get the feature labels if they exist from the SAE
labels = dataset.feature_labels()

# Pass in a feature index to get a more accurate label
new_label = await dataset.label_feature(feature = 65478) # example: "Friendly greetings"

# Annotate a document for a given feature, marking activating tokens with << >>.
annotated_document = dataset[0].token_activations(feature = 65478)

# Extract a list of top documents for a given feature
top_documents = dataset.top_documents_for_feature(feature = 65478)
```

For analyses (e.g. dataset diffing, correlations) done on example datasets, see the `examples/` folder.

## How does this work?

To embed a document, we pass the data into a "reader" LLM and use a sparse autoencoder (SAE) to decompose its internal representation into interpretable concepts known as "features". A SAE produces a sparse, high-dimensional vector (e.g. hundreds of thousands of dimensions) of feature activations per token that we aggregate into a single document embedding.