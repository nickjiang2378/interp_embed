import torch
from abc import ABC, abstractmethod
from enum import Enum
from .utils import ensure_loaded

class BaseSAE(ABC):
  def __init__(self, truncate = True, use_assistant_role: bool = True):
    self.loaded = False
    self.tokenizer = None
    self.truncate = truncate
    self._feature_labels = dict()
    self.use_assistant_role = use_assistant_role

  @classmethod
  def from_metadata(cls, metadata):
    return cls(**metadata)

  def metadata(self):
    return {
      "truncate": self.truncate,
      "use_assistant_role": self.use_assistant_role
    }

  def load(self):
    self.load_models()
    self.loaded = True

  def feature_labels(self):
    return self._feature_labels

  def is_loaded(self):
    return self.loaded

  @ensure_loaded
  def encode_chat(self, chat_conversations):
    texts = [self.tokenizer.apply_chat_template(chat_conversation, tokenize=False) for chat_conversation in chat_conversations]
    return self.encode(texts)

  @ensure_loaded
  def destroy(self):
    self.destroy_models()
    torch.cuda.empty_cache()
    self.loaded = False

  @ensure_loaded
  def chat_template_exists(self):
    return self.tokenizer.chat_template != None

  @ensure_loaded
  def tokenize(self, documents, as_tokens = True, padding: bool = False):
    """
    Tokenizes a list of documents.

    :param documents: List of documents to tokenize. Must be a list of strings
    :param as_tokens: If True, return human-readable token strings. If False, return token IDs
    :param padding: Whether to pad sequences
    """
    if self.chat_template_exists():
      formatted_text = [
          self.tokenizer.apply_chat_template(
              [
                  {
                      "role": "assistant" if self.use_assistant_role else "user",
                      "content": document
                  }
              ],
              tokenize=False
          )
          for document in documents
      ]
    else:
      formatted_text = documents

    inputs = self.tokenizer(
      formatted_text,
      truncation=self.truncate,
      add_special_tokens=not self.chat_template_exists(),
      padding=padding
    )

    input_ids = inputs["input_ids"]

    if not as_tokens:
      return inputs

    # Return human-readable tokens by decoding each token individually
    return [[self.tokenizer.decode([token_id]) for token_id in input_sequence] for input_sequence in input_ids]

  @abstractmethod
  def load_models(self):
    pass

  @abstractmethod
  def encode(self, texts):
    pass

  @abstractmethod
  def destroy_models(self):
    pass

class SAEType(Enum):
    LOCAL = "local"
    GOODFIRE_API = "goodfire_api"
    GOODFIRE = "goodfire"
