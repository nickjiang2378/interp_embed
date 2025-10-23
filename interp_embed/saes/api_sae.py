import asyncio
import os
from dotenv import load_dotenv
from goodfire.api.features.client import AsyncFeaturesAPI
from goodfire import Variant
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer
from typing import List, Dict, Callable

from .base_sae import BaseSAE
from .utils import ensure_loaded, try_to_load_feature_labels
from ..utils.helpers import run_async_in_any_context, log_tqdm_message

load_dotenv()

class ApiSAE(BaseSAE):
  def __init__(self, max_concurrency: int = 8, max_retries: int = 3, base_delay: int = 2.0, **kwargs):
    super().__init__(**kwargs)
    self.max_retries = max_retries
    self.max_concurrency = max_concurrency
    self.base_delay = base_delay
    self.sem = asyncio.Semaphore(max_concurrency)

  def metadata(self):
    parent_metadata = super().metadata()
    parent_metadata.update({
      "max_retries": self.max_retries,
      "base_delay": self.base_delay,
      "max_concurrency": self.max_concurrency
    })
    return parent_metadata

  async def retry_api_with_backoff(self, coroutine_funcs: List[Callable]):
    """
    Retries a list of asynchronous API call coroutines with exponential backoff.

    Each coroutine in `coros` will be executed with up to `self.max_retries` attempts.
    If a coroutine fails, it will be retried after an exponentially increasing delay, starting from `self.base_delay` seconds.
    All coroutines are run concurrently, but concurrency is limited by the asyncio semaphore (`self.sem`).
    Note: Creating too many coroutines at once can consume significant memory; consider batching calls to this method if needed.

    Args:
        coros (List[Coroutine]): A list of coroutine objects representing API calls.

    Returns:
        List[Any]: A list of results from the successfully completed coroutines, in the same order as `coros`.

    Raises:
        Exception: If a coroutine fails after the maximum number of retries, the exception is raised.
    """
    async def worker(coroutine_func):
      for i in range(self.max_retries):
        try:
          async with self.sem:
            matrix = await coroutine_func()
            return matrix
        except Exception as e:
          if i == self.max_retries - 1:
            raise e
          else:
            print(f"Error calling API: {e}")
            await asyncio.sleep(self.base_delay * (2 ** i))
        log_tqdm_message("Testing 123", level="INFO")
    return await asyncio.gather(*[worker(coroutine_func) for coroutine_func in coroutine_funcs])

class GoodfireApiSAE(ApiSAE):
  def __init__(self, variant_name: str, **kwargs):
    super().__init__(**kwargs)

    GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
    assert GOODFIRE_API_KEY is not None, "GOODFIRE_API_KEY is not set"

    self.client = AsyncFeaturesAPI(GOODFIRE_API_KEY)
    self.variant = Variant(variant_name)

  def metadata(self):
    parent_metadata = super().metadata()
    parent_metadata["variant"] = self.variant.base_model
    parent_metadata["use_assistant_role"] = self.use_assistant_role
    return parent_metadata

  def load_models(self):
    # Load the tokenizer
    try:
      self.tokenizer = AutoTokenizer.from_pretrained(self.variant.base_model)
      self._feature_labels = try_to_load_feature_labels(f"goodfire/{self.variant.base_model}.json")
      if self._feature_labels:
        self._feature_labels = {int(key): value for key, value in self._feature_labels.items()} # Convert keys to ints
    except Exception as e:
      raise Exception(f"Failed to load tokenizer for variant {self.variant.base_model}: {e}")

  @ensure_loaded
  def encode(self, texts):
    chat_conversations = [
        [
            {
                "role": "assistant" if self.use_assistant_role else "user",
                "content": text
            }
        ]
        for text in texts
    ]
    return run_async_in_any_context(self.async_encode_chat(chat_conversations))

  @ensure_loaded
  def encode_chat(self, chat_conversations):
    return run_async_in_any_context(self.async_encode_chat(chat_conversations))

  async def async_encode_chat(self, chat_conversations: List[List[Dict[str, str]]]):
    async def get_activations(chat_conversation):
      activations = await self.client.activations(chat_conversation, self.variant)
      return csr_matrix(activations)

    coroutine_funcs = [
        lambda i=i: get_activations(chat_conversations[i])
        for i in range(len(chat_conversations))
    ]
    return await self.retry_api_with_backoff(coroutine_funcs)

  def destroy_models(self):
    pass
