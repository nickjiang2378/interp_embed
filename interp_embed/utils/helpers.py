from tqdm.auto import tqdm
import time
import os, pickle, tempfile
import asyncio
import concurrent.futures

CHAT_TEMPLATE_END_POSITION_TOKENS = 30
CHAT_TEMPLATE_END_POSITION_ACTIVATIONS = 29

def convert_text_to_dict(texts, text_field = "text"):
  """
  Converts a list of text strings to a list of dictionaries with the text as the value.
  """
  return [{text_field: text} for text in texts]

def feature_activation_dict(sample, feature_id, tokenizer):
  activations = sample.latents("all")
  input_text = sample.text_sample
  formatted_text = tokenizer.apply_chat_template([{"role": "assistant", "content": input_text}], tokenize=False)
  input_ids = tokenizer(formatted_text, return_tensors="pt")
  output = []
  for idx in range(CHAT_TEMPLATE_END_POSITION_TOKENS + 1, activations.shape[0]):
    new_token =  tokenizer.decode([input_ids["input_ids"].squeeze()[idx]])
    output.append({
      "token": new_token,
      "activation": activations[idx, feature_id].item()
    })
  return output

def truncate_chat_template_activations(activations, remove_eot_token = False):
  if remove_eot_token:
    return activations[CHAT_TEMPLATE_END_POSITION_ACTIVATIONS + 1:-1]
  else:
    return activations[CHAT_TEMPLATE_END_POSITION_ACTIVATIONS + 1:]

def truncate_chat_template_tokens(tokens):
  return tokens[CHAT_TEMPLATE_END_POSITION_TOKENS + 1:]

def tokenize(text, tokenizer):
  formatted_text = tokenizer.apply_chat_template([{"role": "assistant", "content": text}], tokenize=False)
  input_ids = tokenizer(formatted_text, return_tensors="pt")
  num_input_ids = input_ids["input_ids"].squeeze().shape[0]
  tokens = []

  for idx in range(num_input_ids):
    new_token =  tokenizer.decode([input_ids["input_ids"].squeeze()[idx]])
    tokens.append(new_token)
  return tokens

def activation_dict_to_string(activation_dict):
  output_string = ""
  for item in activation_dict:
    if item["activation"] > 0:
      output_string += f"[token: {item['token']}, activation: {item['activation']}]"
    else:
      output_string += item["token"]
  return output_string

def sets_are_equal(set1, set2):
  return set1 == set2

def log_tqdm_message(message, level="INFO"):
    """Log a message below the progress bar, works in both terminal and Jupyter"""
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {level}: {message}"
    tqdm.write(formatted_msg)

def highlight_activations_as_string(tokens, activations, left_marker, right_marker):
  result = []
  in_highlight = False
  for token, activation in zip(tokens, activations):
      if activation > 0 and not in_highlight:
          result.append(left_marker)
          in_highlight = True
      if activation <= 0 and in_highlight:
          result.append(right_marker)
          in_highlight = False
      result.append(token)
  if in_highlight:
      result.append(">>")
  return "".join(result)

def token_count_as_string(tokens):
  if tokens < 10**3:
    return str(tokens)
  elif tokens < 10**6:
    thousands = tokens // 10**3
    hundreds = (tokens - thousands * 10**3) // 10**2
    return f"{thousands}.{hundreds}k"
  else:
    millions = tokens // 10**6
    return f"{millions}m"

def safe_save_pkl(data, path):
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        # atomic replace
        os.replace(tmp_path, path)
    finally:
        # cleanup if something went wrong before replace
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def safe_load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def dict_astype(dictionary, dtype):
    return {key: value.astype(dtype) for key, value in dictionary.items()}

def has_running_loop():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False

def run_in_new_loop(coro):
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  try:
      return loop.run_until_complete(coro)
  finally:
      loop.close()

def run_async_in_any_context(coro):
    if has_running_loop():
        # Running in async context (like Jupyter)
        # Run in separate thread with new event loop to avoid deadlock
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop, coro)
            return future.result()
    else:
        # Running in sync context, use asyncio.run
        return asyncio.run(coro)


def compute_token_count(rows):
    token_lengths = [len(row.tokenized_document) for row in rows if row is not None]
    return sum(token_lengths) if token_lengths else 0