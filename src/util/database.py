import os
from enum import Enum
from typing import Optional

from db.models import File, FileType, ModelConfig, OffsetConfig, PdfConfig
from util.context import Context

from .hash import generate_file_sha256


def get_file(filename: str) -> File:
    existing_file = File.objects.filter(path=filename).first()
    stats = os.stat(filename)
    size, mtime, ctime = stats.st_size, stats.st_mtime_ns, stats.st_ctime_ns
    if existing_file:
        if (
            existing_file.size != size
            or existing_file.mtime != mtime
            or existing_file.ctime != ctime
        ):
            hash = generate_file_sha256(filename)
            existing_file.size = size
            existing_file.mtime = mtime
            existing_file.ctime = ctime
            existing_file.hash = hash
            existing_file.hash_method = "sha256"
            existing_file.save()
            return existing_file
        else:
            return existing_file
    else:
        hash = generate_file_sha256(filename)
        file = File(
            path=filename,
            size=size,
            mtime=mtime,
            ctime=ctime,
            hash=hash,
            hash_method="sha256",
        )
        file.save()
        return file


class Phase(Enum):
    FILE = 1
    CONTENT = 2
    TOKENS = 3
    EMBEDDINGS = 4


def process_file(ctx: Context, filename: str):
    file = File.get_file(ctx, filename)

    # Check file type
    if file.get_suggested_type() == FileType.PDF:
        # Get PDF transform
        pdf_loader, _ = PdfConfig.get_default_config()
        pdf_transform = pdf_loader.transform(ctx, file)
        text_file = pdf_transform.output_txt_file
    else:
        # Text file
        # TODO: CSV support
        text_file = file

    # Tokenize text file
    print("TOKENIZING")
    model_loader, _ = ModelConfig.get_default_config(ctx)
    token_transform = model_loader.tokenize(ctx, text_file)

    # Embed tokens
    print("EMBEDDING")
    offset_loader, _ = OffsetConfig.get_default_config()
    embedding_transform = model_loader.embed(ctx, token_transform, offset_loader)

    return {
        "text": text_file.read_as_text(),
        "tokens": token_transform.output_token_json_file.read_as_json(),
        "offsets": embedding_transform.get_offsets(),
        "embeddings": embedding_transform.get_embeddings(ctx),
    }


def update_files_in_db(
    ctx: Context, filenames: list[str], max_phase: Optional[Phase] = None
) -> list[str]:
    print("UPDATING", filenames)
    for filename in filenames:
        file = File.get_file(ctx, filename)

        if max_phase == Phase.FILE:
            continue

        # Check file type
        if file.get_suggested_type() == FileType.PDF:
            # Get PDF transform
            pdf_loader, _ = PdfConfig.get_default_config()
            pdf_transform = pdf_loader.transform(ctx, file)
            text_file = pdf_transform.output_txt_file
        else:
            # Text file
            # TODO: CSV support
            text_file = file

        if max_phase == Phase.CONTENT:
            continue

        # Tokenize text file
        print("TOKENIZING")
        model_loader, _ = ModelConfig.get_default_config(ctx)
        token_transform = model_loader.tokenize(ctx, text_file)

        if max_phase == Phase.TOKENS:
            continue

        # Embed tokens
        print("EMBEDDING")
        offset_loader, _ = OffsetConfig.get_default_config()
        embedding_transform = model_loader.embed(ctx, token_transform, offset_loader)

    return filenames
