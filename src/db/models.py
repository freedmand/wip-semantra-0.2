import json
import os
from typing import Optional, Tuple

from django.db import models
from docarray import Document, DocumentArray

from manage import init_django
from util.context import Context
from util.hash import generate_file_sha256
from util.model import BaseModel
from util.pdf import transform_pdf

init_django()

DEFAULT_TRANSFORMERS_MODEL = "sentence-transformers/all-mpnet-base-v2"


class FileType(models.TextChoices):
    PDF = "pdf"
    CSV = "csv"
    TEXT = "txt"


class File(models.Model):
    id = models.AutoField(primary_key=True)
    path = models.CharField(max_length=255, unique=True)
    hash = models.CharField(max_length=255)
    hash_method = models.CharField(max_length=255)
    size = models.IntegerField()
    ctime = models.BigIntegerField()
    mtime = models.BigIntegerField()

    def read_as_text(self) -> str:
        with open(self.path, "r") as f:
            return f.read()

    def read_as_json(self):
        with open(self.path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_file(ctx: Context, filename: str) -> Optional["File"]:
        existing_file = File.objects.filter(path=filename).first()
        try:
            stats = os.stat(filename)
        except Exception as e:
            ctx.log_error(e)
            return None
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

    def get_suggested_type(self):
        path = self.path.lower()
        if path.endswith(".pdf"):
            return FileType.PDF
        elif path.endswith(".csv"):
            return FileType.CSV
        else:
            return FileType.TEXT

    def __str__(self):
        return self.path


class ContentFile(File):
    file_type: FileType = models.CharField(max_length=255, choices=FileType.choices)


class PdfFile(ContentFile):
    page_map = models.JSONField()


# Abstract class
class Config(models.Model):
    id = models.AutoField(primary_key=True)

    @staticmethod
    def get_default_config():
        raise NotImplementedError()

    class Meta:
        abstract = True


class PdfConfig(Config):
    @staticmethod
    def get_default_config():
        return PdfConfig.objects.get_or_create()

    def transform(self, ctx: Context, pdf_file: PdfFile) -> Optional["PdfTransform"]:
        # Check if transform already exists
        pdf_transform = PdfTransform.objects.filter(
            input_pdf_file=pdf_file, pdf_loader=self
        ).first()
        if pdf_transform:
            return pdf_transform

        try:
            output_txt_file = ctx.new_file_name()
            output_page_position_json_file = ctx.new_file_name()
            page_count = transform_pdf(
                ctx, pdf_file.path, output_txt_file, output_page_position_json_file
            )

            pdf_transform = PdfTransform(
                input_pdf_file=pdf_file,
                pdf_loader=self,
                num_pages=page_count,
                output_txt_file=File.get_file(ctx, output_txt_file),
                output_page_position_json_file=File.get_file(
                    ctx, output_page_position_json_file
                ),
            )
            pdf_transform.save()
            return pdf_transform
        except Exception as e:
            # Safely remove files if they exist
            if os.path.exists(output_txt_file):
                os.remove(output_txt_file)
            if os.path.exists(output_page_position_json_file):
                os.remove(output_page_position_json_file)

            ctx.log_error(e)
            return None


# class TokenLoader(Loader):
#     tokenizer = models.CharField(max_length=255, choices=Tokenizer.choices)
#     tokenizer_model = models.CharField(max_length=255)

#     @staticmethod
#     def get_default_loader():
#         return TokenLoader.objects.get_or_create(
#             tokenizer=TokenLoader.Tokenizer.TRANSFORMERS,
#             model=DEFAULT_TRANSFORMERS_MODEL,
#         )

#     def tokenize(self, ctx: Context, text_file: File) -> "TokenTransform":
#         tokenizer = ctx.get_tokenizer(self.tokenizer, self.tokenizer_model)
#         output_token_json_file = ctx.new_file_name()
#         tokens = tokenizer.tokenize(text_file.read_as_text())
#         with open(output_token_json_file, "w") as f:
#             json.dump(tokens, f)

#         token_transform = TokenTransform(
#             input_text_file=text_file,
#             token_loader=self,
#             num_tokens=len(tokens),
#             output_token_json_file=File.get_file(ctx, output_token_json_file),
#         )
#         token_transform.save()
#         return token_transform


class ModelConfig(Config):
    class ModelType(models.TextChoices):
        TRANSFORMERS = "transformers"
        OPENAI = "openai"

    model_type = models.CharField(max_length=255, choices=ModelType.choices)
    model_name = models.CharField(max_length=255)
    num_dimensions = models.IntegerField()

    @staticmethod
    def get_default_config(ctx: Context):
        model = ctx.get_transformers_model(DEFAULT_TRANSFORMERS_MODEL)
        num_dimensions = model.get_num_dimensions()
        return ModelConfig.objects.get_or_create(
            model_type=ModelConfig.ModelType.TRANSFORMERS,
            model_name=DEFAULT_TRANSFORMERS_MODEL,
            num_dimensions=num_dimensions,
        )

    def get_model(self, ctx: Context) -> BaseModel:
        if self.model_type == ModelConfig.ModelType.TRANSFORMERS:
            return ctx.get_transformers_model(self.model_name)
        elif self.model_type == ModelConfig.ModelType.OPENAI:
            return ctx.get_openai_model(self.model_name)
        else:
            raise Exception(f"Unknown model type {self.model_type}")

    def get_annlite_store(self, ctx: Context) -> DocumentArray:
        directory_path = os.path.join(ctx.app_dir, "annlite", f"{self.id}")
        print("DIRECTORY PATH", directory_path)
        os.makedirs(directory_path, exist_ok=True)
        return DocumentArray(
            storage="annlite",
            config={
                "data_path": directory_path,
                "n_dim": self.num_dimensions,
                "metric": "cosine",
            },
        )

    def tokenize(self, ctx: Context, text_file: File) -> "TokenTransform":
        # Check if transform already exists
        token_transform = TokenTransform.objects.filter(
            input_text_file=text_file, model_loader=self
        ).first()
        if token_transform:
            return token_transform

        model = self.get_model(ctx)
        # TODO: figure out strip
        text = text_file.read_as_text().strip()
        raw_tokens = model.get_tokens(text)
        tokens = model.get_text_chunks(text, raw_tokens)
        output_token_json_file = ctx.new_file_name()
        with open(output_token_json_file, "w") as f:
            json.dump(tokens, f)

        token_transform = TokenTransform(
            input_text_file=text_file,
            model_loader=self,
            num_tokens=len(tokens),
            output_token_json_file=File.get_file(ctx, output_token_json_file),
        )
        token_transform.save()
        return token_transform

    def embed(
        self,
        ctx: Context,
        token_transform: "TokenTransform",
        offset_loader: "OffsetConfig",
    ) -> "EmbeddingTransform":
        # Check if transform already exists
        embedding_transform = EmbeddingTransform.objects.filter(
            input_tokens_file=token_transform.output_token_json_file,
            model_loader=self,
            offset_loader=offset_loader,
        ).first()
        if embedding_transform:
            return embedding_transform

        model = self.get_model(ctx)
        text_chunks = token_transform.output_token_json_file.read_as_json()
        offsets = offset_loader.get_offsets(len(text_chunks))

        # Write offsets to json file
        output_offsets_json_file = ctx.new_file_name()
        with open(output_offsets_json_file, "w") as f:
            json.dump(offsets, f)

        # Calculate embeddings
        tokens = model.get_tokens("".join(text_chunks))
        embeddings = model.embed(tokens, offsets)

        # Extend document array annlite store
        docarray = self.get_annlite_store(ctx)
        doc = Document()
        doc.chunks = [Document(embedding=embedding) for embedding in embeddings]
        docarray.extend([doc])

        embedding_transform = EmbeddingTransform(
            input_tokens_file=token_transform.output_token_json_file,
            model_loader=self,
            offset_loader=offset_loader,
            num_embeddings=len(embeddings),
            output_offsets_json_file=File.get_file(ctx, output_offsets_json_file),
            output_embeddings_doc_id=doc.id,
        )
        embedding_transform.save()

        return embedding_transform


class Transform(models.Model):
    id = models.AutoField(primary_key=True)

    @staticmethod
    def get_config_class():
        raise NotImplementedError()

    @staticmethod
    def transform(ctx: Context, config: "Config") -> "Transform":
        raise NotImplementedError()

    class Meta:
        abstract = True


class TokenTransform(Transform):
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["input_text_file", "model_loader"],
                name="unique_token_transform",
            )
        ]

    input_text_file = models.ForeignKey(
        File, on_delete=models.CASCADE, related_name="+"
    )
    model_loader = models.ForeignKey(
        ModelConfig, on_delete=models.CASCADE, related_name="+"
    )
    num_tokens = models.IntegerField()
    output_token_json_file = models.ForeignKey(
        File, on_delete=models.CASCADE, related_name="+"
    )


class OffsetConfig(Config):
    class OffsetType(models.TextChoices):
        CHUNKS = "chunks"

    offset_type = models.CharField(max_length=255, choices=OffsetType.choices)
    offset_size = models.IntegerField()
    overlap_size = models.IntegerField()

    @staticmethod
    def get_default_config():
        return OffsetConfig.objects.get_or_create(
            offset_type=OffsetConfig.OffsetType.CHUNKS,
            offset_size=128,
            overlap_size=16,
        )

    def get_offsets(self, length: int) -> list[Tuple[int, int]]:
        if self.offset_type == OffsetConfig.OffsetType.CHUNKS:
            start = 0
            offsets = []
            while start < length:
                end = min(start + self.offset_size, length)
                offsets.append((start, end))
                start = start + self.offset_size - self.overlap_size
            return offsets
        else:
            raise Exception(f"Unknown offset type {self.offset_type}")


class EmbeddingTransform(Transform):
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["input_tokens_file", "model_loader", "offset_loader"],
                name="unique_embedding_transform",
            )
        ]

    input_tokens_file = models.ForeignKey(
        File, on_delete=models.CASCADE, related_name="+"
    )
    model_loader = models.ForeignKey(
        ModelConfig, on_delete=models.CASCADE, related_name="+"
    )
    offset_loader = models.ForeignKey(
        OffsetConfig, on_delete=models.CASCADE, related_name="+"
    )
    num_embeddings = models.IntegerField()
    output_offsets_json_file = models.ForeignKey(
        File, on_delete=models.CASCADE, related_name="+"
    )
    output_embeddings_doc_id = models.CharField(max_length=255)

    def get_offsets(self) -> list[Tuple[int, int]]:
        return self.output_offsets_json_file.read_as_json()

    def get_embeddings(self, ctx: Context) -> list[list[float]]:
        docarray = self.model_loader.get_annlite_store(ctx)
        doc = docarray[self.output_embeddings_doc_id]
        return [chunk.embedding for chunk in doc.chunks]


class PdfTransform(Transform):
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["input_pdf_file", "pdf_loader"], name="unique_pdf_file_loader"
            )
        ]

    input_pdf_file = models.ForeignKey(File, on_delete=models.CASCADE, related_name="+")
    pdf_loader = models.ForeignKey(
        PdfConfig, on_delete=models.CASCADE, related_name="+"
    )
    num_pages = models.IntegerField()
    output_txt_file = models.ForeignKey(
        File, on_delete=models.CASCADE, related_name="+"
    )
    output_page_position_json_file = models.ForeignKey(
        File, on_delete=models.CASCADE, related_name="+"
    )


# class CsvLoader(Loader):
#     field_indices


# class PdfContent(models.Model):
#     source_file = models.ForeignKey(File, on_delete=models.CASCADE)
#     content_file = models.ForeignKey(
#         File, on_delete=models.CASCADE, related_name="content_file"
#     )
