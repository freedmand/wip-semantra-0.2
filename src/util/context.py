import os
import uuid

import click
from ilock import ILock
from tqdm import tqdm

from util.model import BaseModel, OpenAIModel, TransformersModel

APP_NAME = "semantra3"
APP_DIR = click.get_app_dir(APP_NAME)


class Context:
    def __init__(self, app_dir=APP_DIR, silent=False):
        self.app_dir = app_dir
        self.silent = silent
        self.model_cache: dict[str, BaseModel] = {}
        self.setup()

    def get_transformers_model(self, model_name: str) -> TransformersModel:
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        else:
            with ILock(f"transformers_{model_name}"):
                model = TransformersModel(model_name)
                self.model_cache[model_name] = model
                return model

    def get_openai_model(self, model_name: str) -> OpenAIModel:
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        else:
            with ILock(f"openai_{model_name}"):
                model = OpenAIModel(model_name)
                self.model_cache[model_name] = model
                return model

    def setup(self):
        if not os.path.exists(self.app_dir):
            os.makedirs(self.app_dir)

    def log_error(self, error: Exception):
        print("ERROR LOGGED FROM CTX", error)

    def progress(self, iterable, **kwargs):
        return tqdm(iterable, leave=False, disable=self.silent, **kwargs)

    def new_file_name(self):
        path = os.path.join(self.app_dir, str(uuid.uuid4()))
        return path
