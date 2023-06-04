import os
from abc import ABC, abstractmethod

import numpy as np
import openai
import tiktoken
import torch
from dotenv import load_dotenv
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoModel, AutoTokenizer

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

if "OPENAI_API_KEY" in os.environ:
    openai.api_key = os.getenv("OPENAI_API_KEY")


minilm_model_name = "sentence-transformers/all-MiniLM-L6-v2"
mpnet_model_name = "sentence-transformers/all-mpnet-base-v2"
sgpt_model_name = "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit"
sgpt_1_3B_model_name = "Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit"
instructor_base_model_name = "hkunlp/instructor-base"
instructor_large_model_name = "hkunlp/instructor-large"
instructor_xlarge_model_name = "hkunlp/instructor-xl"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def filter_none(x):
    return [i for i in x if i is not None]


def as_numpy(x):
    # If x is a tensor, convert it to a numpy array
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


class BaseModel(ABC):
    @abstractmethod
    def get_num_dimensions(self) -> int:
        ...

    @abstractmethod
    def get_tokens(self, text: str):
        ...

    @abstractmethod
    def get_token_length(self, tokens) -> int:
        ...

    @abstractmethod
    def concat_tokens(self, tokens1, tokens2):
        ...

    @abstractmethod
    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        ...

    @abstractmethod
    def get_config(self):
        ...

    @abstractmethod
    def embed(self, tokens, offsets, is_query: bool = False) -> "list[list[float]]":
        ...

    @abstractmethod
    def is_asymmetric(self) -> bool:
        ...

    def embed_document(self, document) -> "list[float]":
        tokens = self.get_tokens(document)
        return self.embed(tokens, [(0, self.get_token_length(tokens))], False)[0]

    def embed_query(self, query: str) -> "list[float]":
        tokens = self.get_tokens(query)
        return self.embed(tokens, [(0, self.get_token_length(tokens))], True)[0]

    def embed_queries(self, queries) -> "list[float]":
        all_embeddings = [
            as_numpy(self.embed_query(query["query"])) * query["weight"]
            for query in queries
        ]
        # Return sum of embeddings
        return np.sum(all_embeddings, axis=0)

    def embed_queries_and_preferences(self, queries, preferences, documents):
        query_embedding = self.embed_queries(queries) if len(queries) > 0 else None
        # Add preferences to embeddings
        return np.sum(
            [
                *([query_embedding] if query_embedding is not None else []),
                *[
                    documents[pref["file"]["filename"]].embeddings[
                        pref["searchResult"]["index"]
                    ]
                    * pref["weight"]
                    for pref in preferences
                ],
            ],
            axis=0,
        )

    def is_asymmetric(self):
        return False


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_name="text-embedding-ada-002",
        num_dimensions=1536,
        tokenizer_name="cl100k_base",
    ):
        # Check if OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            raise Exception(
                "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable or create a `.env` file with the key in the same directory."
            )
        self.model_name = model_name
        self.num_dimensions = num_dimensions
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def get_config(self):
        return {
            "model_type": "openai",
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer.name,
        }

    def get_num_dimensions(self) -> int:
        return self.num_dimensions

    def get_tokens(self, text: str):
        return self.tokenizer.encode(text)

    def get_token_length(self, tokens) -> int:
        return len(tokens)

    def concat_tokens(self, tokens1, tokens2):
        return tokens1 + tokens2

    def get_text_chunks(self, _: str, tokens) -> "list[str]":
        return [self.tokenizer.decode([token]) for token in tokens]

    def embed(self, tokens, offsets, _is_query=False) -> "list[list[float]]":
        texts = [tokens[i:j] for i, j in offsets if tokens[i:j] != []]
        response = openai.Embedding.create(model=self.model_name, input=texts)
        return np.array([data["embedding"] for data in response["data"]])


def zero_if_none(x):
    return 0 if x is None else x


class TransformersModel(BaseModel):
    def __init__(
        self,
        model_name,
        doc_token_pre=None,
        doc_token_post=None,
        query_token_pre=None,
        query_token_post=None,
        asymmetric=False,
        cuda=None,
    ):
        if cuda is None:
            cuda = torch.cuda.is_available()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Get tokens
        self.pre_post_tokens = [
            doc_token_pre,
            doc_token_post,
            query_token_pre,
            query_token_post,
        ]
        self.doc_token_pre = (
            self.tokenizer.encode(doc_token_pre, add_special_tokens=False)
            if doc_token_pre
            else None
        )
        self.doc_token_post = (
            self.tokenizer.encode(doc_token_post, add_special_tokens=False)
            if doc_token_post
            else None
        )
        self.query_token_pre = (
            self.tokenizer.encode(query_token_pre, add_special_tokens=False)
            if query_token_pre
            else None
        )
        self.query_token_post = (
            self.tokenizer.encode(query_token_post, add_special_tokens=False)
            if query_token_post
            else None
        )

        self.asymmetric = asymmetric

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def get_config(self):
        return {
            "model_type": "transformers",
            "model_name": self.model_name,
            "doc_token_pre": self.pre_post_tokens[0],
            "doc_token_post": self.pre_post_tokens[1],
            "query_token_pre": self.pre_post_tokens[2],
            "query_token_post": self.pre_post_tokens[3],
            "asymmetric": self.asymmetric,
        }

    def is_asymmetric(self):
        return self.asymmetric

    def get_num_dimensions(self) -> int:
        return int(self.model.config.hidden_size)

    def get_tokens(self, text: str):
        return self.tokenizer(
            text, return_offsets_mapping=True, verbose=False, return_tensors="pt"
        )

    def get_token_length(self, tokens) -> int:
        return len(tokens["input_ids"][0])

    def concat_tokens(self, tokens1, tokens2):
        return {
            "input_ids": torch.cat([tokens1["input_ids"], tokens2["input_ids"]], dim=1),
            "attention_mask": torch.cat(
                [tokens1["attention_mask"], tokens2["attention_mask"]], dim=1
            ),
            "offset_mapping": torch.cat(
                [tokens1["offset_mapping"], tokens2["offset_mapping"]], dim=1
            ),
        }

    def get_text_chunks(self, text: str, tokens) -> "list[str]":
        offsets = tokens["offset_mapping"][0]
        chunks = []
        prev_i = None
        prev_j = None
        for i, j in offsets:
            new_i = prev_j if i == j else i
            if prev_i is not None:
                chunks.append(text[prev_i:new_i])
            if prev_i is None:
                prev_i = 0
            elif new_i > prev_i:
                prev_i = new_i
            if prev_j is None:
                prev_j = j
            elif j > prev_j:
                prev_j = j
        chunks.append(text[0 if prev_i is None else prev_i :])
        return chunks

    def normalize_input_ids(self, input_ids, is_query):
        if self.query_token_pre is None and self.query_token_post is None:
            return input_ids
        else:
            token_pre = self.query_token_pre if is_query else self.doc_token_pre
            token_post = self.query_token_post if is_query else self.doc_token_post
            return torch.cat(
                filter_none(
                    [
                        torch.tensor(token_pre) if token_pre is not None else None,
                        input_ids,
                        torch.tensor(token_post) if token_post is not None else None,
                    ]
                )
            )

    def normalize_attention_mask(self, attention_mask, is_query):
        if self.query_token_pre is None and self.query_token_post is None:
            return attention_mask
        else:
            token_pre = self.query_token_pre if is_query else self.doc_token_pre
            token_post = self.query_token_post if is_query else self.doc_token_post
            return torch.cat(
                filter_none(
                    [
                        torch.ones(len(token_pre)) if token_pre is not None else None,
                        attention_mask,
                        torch.ones(len(token_post)) if token_post is not None else None,
                    ]
                )
            )

    def embed(self, tokens, offsets, is_query=False) -> "list[list[float]]":
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_input_ids(
                    tokens["input_ids"][0].index_select(0, torch.tensor(range(i, j))),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=zero_if_none(self.tokenizer.pad_token_id),
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [
                self.normalize_attention_mask(
                    tokens["attention_mask"][0].index_select(
                        0, torch.tensor(range(i, j))
                    ),
                    is_query,
                )
                for i, j in offsets
            ],
            batch_first=True,
            padding_value=0,
        )
        if self.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            model_output = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        return mean_pooling(model_output, attention_mask)


class InstructorModel(TransformersModel):
    def __init__(self, model_name, doc_instruction, query_instruction, cuda=None):
        self.model_name = model_name
        self.doc_instruction = doc_instruction
        self.query_instruction = query_instruction
        self.model = INSTRUCTOR(model_name)
        self.tokenizer = self.model.tokenizer

    def get_num_dimensions(self) -> int:
        return self.model.encode("").shape[0]

    def get_tokens(self, text: str):
        return self.model.tokenizer(
            text, return_offsets_mapping=True, verbose=False, return_tensors="pt"
        )

    def get_config(self):
        return {
            "model_type": "instructor",
            "model_name": self.model_name,
            "doc_instruction": self.doc_instruction,
            "query_instruction": self.query_instruction,
        }

    def embed(self, tokens, offsets, is_query: bool = False) -> list[list[float]]:
        instruction = self.query_instruction if is_query else self.doc_instruction
        texts = [
            [
                instruction,
                self.tokenizer.decode(
                    tokens["input_ids"][0][offset[0] : offset[1]],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            ]
            for offset in offsets
        ]
        return self.model.encode(texts)


def get_default(kwargs, field, default):
    value = kwargs.get(field, default)
    if value is None:
        return default
    else:
        return value


models = {
    "openai": {
        "cost_per_token": 0.0004 / 1000,
        "pool_size": 50000,
        "pool_count": 2000,
        "get_model": lambda **kwargs: OpenAIModel(
            model_name="text-embedding-ada-002",
            num_dimensions=1536,
            tokenizer_name="cl100k_base",
        ),
    },
    "minilm": {
        "cost_per_token": None,
        "pool_size": 50000,
        "get_model": lambda **kwargs: TransformersModel(model_name=minilm_model_name),
    },
    "mpnet": {
        "cost_per_token": None,
        "pool_size": 15000,
        "get_model": lambda **kwargs: TransformersModel(model_name=mpnet_model_name),
    },
    "sgpt": {
        "cost_per_token": None,
        "pool_size": 10000,
        "get_model": lambda **kwargs: TransformersModel(
            model_name=sgpt_model_name,
            query_token_pre="[",
            query_token_post="]",
            doc_token_pre="{",
            doc_token_post="}",
            asymmetric=True,
        ),
    },
    "sgpt-1.3B": {
        "cost_per_token": None,
        "pool_size": 1000,
        "get_model": lambda **kwargs: TransformersModel(
            model_name=sgpt_1_3B_model_name,
            query_token_pre="[",
            query_token_post="]",
            doc_token_pre="{",
            doc_token_post="}",
            asymmetric=True,
        ),
    },
    "instructor-base": {
        "cost_per_token": None,
        "pool_size": 10000,
        "get_model": lambda **kwargs: InstructorModel(
            model_name=instructor_base_model_name,
            doc_instruction=get_default(
                kwargs, "doc_instruction", "Represent the document for retrieval"
            ),
            query_instruction=get_default(
                kwargs,
                "query_instruction",
                "Represent the query for retrieving supporting documents",
            ),
        ),
    },
}
