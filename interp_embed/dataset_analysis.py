from tqdm.auto import tqdm
import uuid
import numpy as np
import pickle
import pandas as pd
import math
from scipy.sparse import csr_matrix, vstack
from .utils.helpers import (
    tokenize,
    truncate_chat_template_activations,
    truncate_chat_template_tokens,
    highlight_activations_as_string,
    log_tqdm_message,
    safe_save_pkl,
    safe_load_pkl,
    dict_astype,
    compute_token_count,
)

from .saes.load_sae import load_sae_from_metadata
from .llm.utils import get_llm_client, call_async_llm, extract_json_from_response
from .llm.prompts import build_labeling_prompt, build_scoring_prompt
from .utils.data_models import FeatureLabelResponse, SingleSampleScoringResponse
import random
import json
import asyncio

SAMPLE_TRUNCATION_LENGTH = 100

class Dataset():
    def __init__(
        self,
        data,
        sae,
        dataset_description="",
        rows = None,
        field="text",
        compute_activations=True,
        feature_labels=None,
        save_path=None,
        save_every_batch=5,
        batch_size = 8,
    ):
        """
        Initialize a Dataset instance. Computes feature activations over the column marked by with `field`

        :param data: Pandas dataframe with text to get feature activations on specified under `field`
        :param sae: Subclass of BaseSAE. Used to compute feature activations, tokenize data, and provide feature labels.
        :param dataset_description: Optional description of the dataset
        :param field: Field name containing the text data
        :param feature_activations: Optional pre-computed feature activations. If None, will compute them. Maps sae id to a sparse matrix.
        :param save_path: Optional file path to save dataset feature activations when computing. Allows for recovery if dataset creation fails.
        """
        if isinstance(data, pd.DataFrame):
            data_list = data.to_dict(orient = "records")
        else:
            data_list = data

        assert isinstance(data_list, list), "data must be a list"
        if len(data_list) > 0:
            assert isinstance(data_list[0], dict), "data must be a list of dictionaries"
            assert field in data_list[0], f"field {field} not found in data"

        self.id = str(uuid.uuid4())[:6]
        self.dataset_description = dataset_description
        self._feature_labels = feature_labels or dict()

        # Store the dataset as a DataFrame
        self.dataset = pd.DataFrame(data_list)
        self.num_documents = len(self.dataset)
        self.field = field
        self.sae = sae
        self.rows = rows or [None] * self.num_documents # Initialize rows with None
        self.token_count = compute_token_count(self.rows)
        document_list = [row[field] for row in data_list]
        # Preprocessing on document list

        # Compute feature activations on the document list
        if compute_activations:
            self._compute_latents(save_path, save_every_batch, batch_size=batch_size)

    def _compute_latents(self, save_path = None, save_every_batch = 5, batch_size = 8):

        data_as_dict = self.dataset.to_dict(orient = "records")
        document_list = [row[self.field] for row in data_as_dict]

        # Find remaining work to do
        selected_document_indices = []
        for row_id in range(len(self.rows)):
            if self.rows[row_id] == None:
                selected_document_indices.append(row_id)

        if len(selected_document_indices) == 0:
            return

        print(f"Found {len(selected_document_indices)} rows with empty or incomplete feature activations")

        if not self.sae.is_loaded():
            self.sae.load()

        self._feature_labels = self._feature_labels or self.sae.feature_labels() or dict()

        pbar = tqdm(range(math.ceil(len(selected_document_indices) / batch_size)),
           desc="Computing latents")
        for i in pbar:
            selected_documents = [document_list[ind] for ind in selected_document_indices[batch_size*i: batch_size*(i + 1)]]
            try:
                # Compute latents for the batch of documents
                batch_activations = self.sae.encode(selected_documents)

                # Tokenize the documents
                tokenized_documents = self.sae.tokenize(selected_documents)

                # Update the successful token count
                self.token_count += sum([activations.shape[0] for activations in batch_activations if activations is not None])
            except Exception as e:
                log_tqdm_message(f"ERROR (batch {i}): {e}")
                batch_activations, tokenized_documents = [None] * len(selected_documents), None

            for j, batch_activation in enumerate(batch_activations):
                doc_index = selected_document_indices[batch_size*i + j]
                if batch_activation != None:
                    new_row = DatasetRow(row=data_as_dict[doc_index], tokenized_document=tokenized_documents[j], field=self.field, activations = batch_activation)
                    self.rows[doc_index] = new_row
            if save_path and (i + 1) % save_every_batch == 0:
                self.save_to_file(save_path)
            pbar.set_description(f"Computing latents \u2022 {self.token_count} tokens")
        self.sae.destroy() # Remove the language model and SAE from memory
        if save_path:
            self.save_to_file(save_path)

    def save_to_file(self, file_path = None, dtype=np.float32):
        """
        Save the Dataset parameters to a file.

        :param file_path: Path to the file where the parameters will be saved
        """
        file_path = file_path or f"dataset_{self.id}.pkl"

        # Save only the essential parameters, keeping activations in sparse format
        save_dict = {
            'dataset': self.dataset,
            'rows': [row.latents("all", compress=True).astype(dtype) if row else None for row in self.rows],  # Keep as csr_matrix
            'aggregate_activations': [
                dict_astype(row.aggregate_activations, dtype) if row else None
                for row in self.rows
            ],
            'tokenized_documents': [row.tokenized_document if row else None for row in self.rows],
            'field': self.field,
            'dataset_description': self.dataset_description,
            'id': self.id,
            'sae_metadata': self.sae.metadata(),
            'feature_labels': self.feature_labels()
        }
        safe_save_pkl(save_dict, file_path)

    @classmethod
    def load_from_file(cls, file_path, resume = False, batch_size = 8):
        """
        Load a Dataset from saved parameters.

        :param file_path: Path to the file containing the saved parameters
        :return: Dataset instance
        """
        params = safe_load_pkl(file_path)

        # Create DatasetRow objects from the saved activations (already in sparse format)
        rows = []
        for i, activation in enumerate(params['rows']):
            if activation == None:
                rows.append(None)
            else:
                rows.append(DatasetRow(
                    row=params['dataset'].iloc[i].to_dict(),
                    field=params['field'],
                    tokenized_document=params['tokenized_documents'][i],
                    activations=activation,
                    aggregate_activations=params['aggregate_activations'][i],
                ))

        # Create and return the Dataset
        dataset = cls(
            data=params['dataset'],
            sae=load_sae_from_metadata(params['sae_metadata']),
            dataset_description=params['dataset_description'],
            rows=rows,
            field=params['field'],
            save_path=file_path if resume else None,
            compute_activations=resume,
            feature_labels=params['feature_labels'],
            batch_size=batch_size,
        )
        dataset.id = params['id']
        return dataset

    def feature_labels(self):
        return {int(key): value for key, value in self._feature_labels.items()}

    def latents(self, aggregation_method = "max", compress = False, activated_threshold = 0):
        """
        Get the feature activations for all samples.

        :param feature_activation_type: Method of aggregating activations across tokens per sample ('max', 'mean', or 'sum')
        :return: Numpy array of feature activations
        """
        if self.num_documents == 0:
            return None

        d_sae = 4096 # default SAE dimension if no latents have been computed
        for row in self.rows:
            if row is not None:
                d_sae = row.latents(compress = True).shape[1]

        if aggregation_method == "all":
            all_activations = []
            for row in self.rows:
                if row is not None:
                    all_activations.append(row.latents("all", compress = compress))
                else:
                    all_activations.append(np.full(d_sae, np.nan) if not compress else csr_matrix(np.full(d_sae, np.nan)))
            return np.array(all_activations, dtype=np.object_) if not compress else all_activations

        all_feature_activations = []
        if aggregation_method in ["max", "mean", "sum", "binarize", "count"]:
            for row in self.rows:
                if row is not None:
                    all_feature_activations.append(row.latents(aggregation_method, compress = True, activated_threshold = activated_threshold))
                else:
                    all_feature_activations.append(np.full(d_sae, np.nan) if not compress else csr_matrix(np.full(d_sae, np.nan)))
        else:
            raise ValueError(f"Unsupported aggregation method for feature activations: {aggregation_method}")

        all_feature_activations_NF = vstack(all_feature_activations)
        return all_feature_activations_NF if compress else all_feature_activations_NF.toarray()

    def top_documents_for_feature(self, feature, aggregation_type = "max", document_only = True, k = 10, select_top = True, include_nonactive_samples = False):
        sorted_dataset = self.sort_by_features([feature], aggregation_type = aggregation_type, include_nonactive_samples = include_nonactive_samples)

        selected_dataset = sorted_dataset[:k] if select_top else sorted_dataset[-k:]

        return [row.token_activations(feature) for row in selected_dataset] if document_only else selected_dataset

    async def score_feature(self, feature, label, model = "google/gemini-2.5-flash", positive_dataset = None, negative_dataset = None, k = 10):
        positive_dataset, negative_dataset = positive_dataset or self, negative_dataset or self

        # Select k indices randomly from 0 to 3k
        superset_positive_samples = positive_dataset.top_documents_for_feature(feature, select_top = True, k = 3 * k)
        indices = random.sample(list(range(len(superset_positive_samples))), k) if len(superset_positive_samples) > k else list(range(len(superset_positive_samples)))
        positive_samples = [superset_positive_samples[i] for i in indices]

        superset_negative_samples = negative_dataset.top_documents_for_feature(feature, select_top = False, k = 3 * k, include_nonactive_samples = True)
        indices = random.sample(list(range(len(superset_negative_samples))), k) if len(superset_negative_samples) > k else list(range(len(superset_negative_samples)))
        negative_samples = [superset_negative_samples[i] for i in indices]

        positive_prompts = [build_scoring_prompt(label, positive_sample, sample_type = "positive") for positive_sample in positive_samples]
        negative_prompts = [build_scoring_prompt(label, negative_sample, sample_type = "negative") for negative_sample in negative_samples]

        # Call API on all prompts in parallel
        llm_client = get_llm_client(is_openai_model=model.startswith("openai/"), is_async=True)
        positive_tasks = [
            call_async_llm(client=llm_client, model=model, messages=[{"role": "user", "content": prompt}])
            for prompt in positive_prompts
        ]
        negative_tasks = [
            call_async_llm(client=llm_client, model=model, messages=[{"role": "user", "content": prompt}])
            for prompt in negative_prompts
        ]

        positive_responses = await asyncio.gather(*positive_tasks)
        negative_responses = await asyncio.gather(*negative_tasks)

        tally = 0
        total = 0
        results = []
        for response in positive_responses + negative_responses:
            try:
                content = response.choices[0].message.content

                # Extract JSON from response
                json_str = extract_json_from_response(content)
                response_data = json.loads(json_str)
                scored_response = SingleSampleScoringResponse(**response_data)
                results.append(scored_response)
                tally += scored_response.score
                total += 1
            except Exception as e:
                results.append(e)

        return {
            "score": tally / total if total > 0 else 0,
            "total_count": total,
            "responses": results,
        }

    async def label_feature(self, feature, model = "google/gemini-2.5-flash", label_and_score = None, positive_dataset = None, negative_dataset = None, k = 20):
        positive_dataset, negative_dataset = positive_dataset or self, negative_dataset or self

        superset_positive_samples = positive_dataset.top_documents_for_feature(feature, select_top = True, k = 3 * k)
        indices = random.sample(list(range(len(superset_positive_samples))), k) if len(superset_positive_samples) > k else list(range(len(superset_positive_samples)))
        positive_samples = [superset_positive_samples[i] for i in indices]

        superset_negative_samples = negative_dataset.top_documents_for_feature(feature, select_top = False, k = 3 * k, include_nonactive_samples = True)
        indices = random.sample(list(range(len(superset_negative_samples))), k) if len(superset_negative_samples) > k else list(range(len(superset_negative_samples)))
        negative_samples = [superset_negative_samples[i] for i in indices]

        labeling_prompt = build_labeling_prompt(positive_samples, negative_samples, label_and_score = label_and_score)
        llm_client = get_llm_client(is_openai_model=model.startswith("openai/"), is_async=True)
        response = await call_async_llm(client=llm_client, model=model, messages=[{"role": "user", "content": labeling_prompt}])

        # Process LLM response
        try:
            content = response.choices[0].message.content

            # Extract JSON from response
            json_str = extract_json_from_response(content)
            response_data = json.loads(json_str)
            feature_label = FeatureLabelResponse(**response_data)
            return feature_label
        except Exception as e:
            print(f"Failed to process LLM response: {e}")

    def token_activations(self, feature):
        return [row.token_activations(feature) for row in self.rows]

    def sort_by_columns(self, columns, descending = True):
        """
        Sort data samples by specified columns.
        """
        df_sorted = self.dataset.sort_values(by = columns, ascending = not descending)
        return self[np.array(df_sorted.index)]

    def sort_by_features(self, features, aggregation_type = "max", descending = True, include_top_feature = True, include_nonactive_samples = True):
        """
        Sort data samples by specified features and activation type.

        """
        feature_activations_DF = self.latents(aggregation_type)

        selected_feature_activations = feature_activations_DF[:, features]
        feature_labels = self.feature_labels()
        top_features = []
        irrelevant_samples = []
        for i, activation in enumerate(selected_feature_activations):
            nonzero_indices = np.nonzero(activation)[0]

            top_feature_index = nonzero_indices[0] if len(nonzero_indices) > 0 else -1
            if top_feature_index != -1:
                top_features.append([self.rows[i].row_record(), top_feature_index, activation[top_feature_index], feature_labels.get(features[top_feature_index], ""), i])
            else:
                irrelevant_samples.append([self.rows[i].row_record(), float("inf"), 0, "NA", i])
        sorted_entries = sorted(
            top_features,
            key=lambda x: (x[1], x[2]),
            reverse = descending
        )
        if include_nonactive_samples:
            if descending:
                sorted_entries = sorted_entries + irrelevant_samples
            else:
                sorted_entries = irrelevant_samples + sorted_entries
        for entry in sorted_entries:
            entry[1] = features[entry[1]] if entry[1] != float("inf") else -1
            if include_top_feature:
                entry[0]["feature_activation"] = entry[2]
                entry[0]["top_feature_label"] = entry[3]
                entry[0]["top_feature_index"] = entry[1]
                self.rows[entry[4]].row["feature_activation"] = entry[2]
                self.rows[entry[4]].row["top_feature_label"] = entry[3]
                self.rows[entry[4]].row["top_feature_index"] = entry[1]

        sorted_data = [entry[0] for entry in sorted_entries]
        sorted_dataset_rows = [self.rows[entry[4]] for entry in sorted_entries]
        return Dataset(sorted_data, self.sae, self.dataset_description, rows=sorted_dataset_rows, feature_labels=self.feature_labels(), field=self.field) # TODO: columns of dataset rows won't be updated.

    def filter_na_rows(self):
        selected_indices = [ind for ind in range(len(self.rows)) if self.rows[ind] is not None]
        return self[selected_indices]

    def dataset_rows(self):
        return self.rows

    def pandas(self):
        return self.dataset

    def list(self):
        return self.dataset.to_dict(orient = "records")

    def documents(self):
        return self.pandas()[self.field].tolist()

    @property
    def columns(self):
        return self.dataset.columns.tolist()

    def __repr__(self):
        if self.num_documents == 0:
            return "Dataset(<EMPTY>)"
        elif self.num_documents < 5:
            rows = [f"{' ' * 4}{i}: {row}" for i, row in enumerate(self.rows)]
        else:
            rows = (
                [f"{' ' * 4}{i}: {row}" for i, row in enumerate(self.rows[:2])] +
                [f"{' ' * 4}..."] +
                [f"{' ' * 4}{len(self.rows) - 2 + i}: {row}" for i, row in enumerate(self.rows[-2:])]
            )
        columns = self.dataset.columns.tolist()
        columns_with_quotes = [f"'{column}'" for column in columns]

        return f"Dataset {self.id}(\n" + f"{' ' * 2}columns=[{', '.join(columns_with_quotes)}]\n" + f"{' ' * 2}rows=[\n" + "\n".join(rows) + "\n  ]" + "\n)"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.rows[index]
        elif isinstance(index, slice):
            selected_data = self.dataset.iloc[index]
            selected_activations = self.rows[index]
            new_dataset = Dataset(selected_data, sae=self.sae, dataset_description=self.dataset_description, rows=selected_activations, feature_labels=self.feature_labels(), field=self.field)
            return new_dataset
        else:
            if isinstance(index, list):
                index = np.array(index)
            if isinstance(index, (np.ndarray, pd.Series)) and (isinstance(index, np.ndarray) and index.dtype == bool or isinstance(index, pd.Series) and index.dtype == bool):
                selected_data = self.dataset[index]
                true_indices = index.index[index] if isinstance(index, pd.Series) else np.where(index)[0]
                selected_activations = [self.rows[i] for i in true_indices]
                new_dataset = Dataset(selected_data, sae=self.sae, dataset_description=self.dataset_description, rows=selected_activations, feature_labels=self.feature_labels(), field=self.field)
                return new_dataset
            elif isinstance(index, np.ndarray) and index.dtype == int:
                selected_data = self.dataset.iloc[index]
                selected_activations = [self.rows[i] for i in index]
                new_dataset = Dataset(selected_data, sae=self.sae, dataset_description=self.dataset_description, rows=selected_activations, feature_labels=self.feature_labels(), field=self.field)
                return new_dataset
            else:
                raise TypeError(f"Indexing with {type(index)} is not supported")

    def __iter__(self):
        return iter(self.rows)

class DatasetRow():
    def __init__(self, row, tokenized_document, activations, truncate_chat_template = False, aggregate_activations = None, field = "text", low_memory = False):
        """
        Initialize a DatasetRow instance.

        :param sample: Dictionary containing the text sample
        :param activations: Optional precomputed token activations (compressed sparse matrix)
        :param aggregate_activations: Optional precomputed aggregate activations (dense matrix)
        :param field: Field in the sample dictionary that contains the text
        """
        assert isinstance(row, dict), f"sample must be a dictionary. Found type {type(row)}"
        assert isinstance(activations, csr_matrix), f"activations must be a scipy.sparse.csr_matrix. Found type {type(activations)}"
        assert field in row, f"field {field} not found in row"
        assert len(tokenized_document) == activations.shape[0], f"Number of tokens must match number of feature activation vectors, {len(tokenized_document)} != {activations.shape[0]}"
        assert len(tokenized_document) > 0, "Empty documents not allowed!"
        self.data = row[field]
        self.field = field
        self.row = row
        self.tokenized_document = tokenized_document
        self.aggregate_activations = dict()
        self.activations = activations
        self.truncate_chat_template = truncate_chat_template

        if not low_memory and aggregate_activations is None:
            truncated_activations = truncate_chat_template_activations(activations.toarray(), remove_eot_token = True) if truncate_chat_template else activations.toarray()
            self.aggregate_activations["max"] = csr_matrix(truncated_activations.max(axis = 0))
            self.aggregate_activations["sum"] = csr_matrix(truncated_activations.sum(axis = 0))
        elif not low_memory:
            self.aggregate_activations = aggregate_activations
        self.n_tokens = activations.shape[0]


    def row_record(self):
        return self.row

    def latents(self, activation_type = "max", compress = False, activated_threshold = 0):
        if activation_type == "all":
            latents = self.activations
        elif activation_type in ["mean", "max", "sum", "binarize", "count"]:
            if activation_type == "mean":
                latents = self.aggregate_activations["sum"] / self.n_tokens
            elif activation_type == "binarize":
                latents = self.aggregate_activations["max"].copy()
                latents.data = (latents.data > activated_threshold).astype(latents.data.dtype)
            elif activation_type == "count":
                all_activations = self.activations.copy()
                all_activations.data = (all_activations.data > activated_threshold).astype(all_activations.data.dtype)
                latents = csr_matrix(all_activations.getnnz(axis=0))
            else:
                latents = self.aggregate_activations[activation_type]
        else:
            raise ValueError(f"Unsupported activation aggregation method: {activation_type}")

        return latents if compress else latents.toarray()

    def token_activations(self, feature, as_string = True, left_marker = "<<", right_marker = ">>"):
        tokens = truncate_chat_template_tokens(self.tokenized_document) if self.truncate_chat_template else self.tokenized_document
        feature_activations = truncate_chat_template_activations(self.latents("all")) if self.truncate_chat_template else self.latents("all")

        activations = feature_activations[:, feature]
        if as_string:
            return highlight_activations_as_string(tokens, activations, left_marker, right_marker)
        else:
            return [{"token": token, "activation": activation.item()} for token, activation in zip(tokens, activations)]

    def document(self):
        return self.data

    def __repr__(self):
        return f"DatasetRow('{self.data[:SAMPLE_TRUNCATION_LENGTH] + '...' if len(self.data) > SAMPLE_TRUNCATION_LENGTH else self.data}')"