import os
from typing import Generator


def get_files_recursive(paths: list[str]) -> Generator[str, None, None]:
    for path in paths:
        if os.path.isfile(path):
            # Append files
            yield path
        elif os.path.isdir(path):
            # Append all files of directories
            for root, _, files in os.walk(path):
                for file in files:
                    if os.path.isfile(os.path.join(root, file)):
                        yield os.path.join(root, file)
