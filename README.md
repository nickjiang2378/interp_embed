# InterpEmbed

`interp_embed` is a toolkit for analyzing unstructured (ex. text) datasets with sparse autoencoders (SAEs). It can quickly compute and efficiently store feature activations for data analysis. Given a dataset of documents, `interp_embed` creates sparse, high-dimensional, interpretable embeddings, where each dimension maps to a concept like syntax or topic, for a variety of downstream analysis tasks like dataset diffing, concept correlations, and directed clustering.

## Setup

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
from interp_embed.saes import GoodfireSAE
import pandas as pd

# 1. Load a Goodfire SAE or SAE supported through the SAELens package
sae = GoodfireSAE(
    variant_name="Llama-3.1-8B-Instruct-SAE-l19",  # or "Llama-3.3-70B-Instruct-SAE-l50" for higher quality features
    device="cuda:0", # optional
    quantize=True # optional
)

# 2. Prepare your data as a DataFrame
df = pd.DataFrame({
    "text": ["Good morning!", "Hello there!", "Good afternoon."],
    "date": ["2022-01-10", "2021-08-23", "2023-03-14"] # Metadata column
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

To embed a document, we pass the data into a "reader" LLM and use a sparse autoencoder (SAE) to decompose its internal representation into interpretable concepts known as "features". The number of features per SAE varies from 1000 - 100000. A SAE produces a sparse, high-dimensional vector of feature activations per token that we aggregate into a single document embedding.