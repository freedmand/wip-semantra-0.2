import os
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext
from torch import Generator

from db.models import EmbeddingTransform, File, PdfTransform, TokenTransform
from util.context import Context
from util.database import Phase, update_files_in_db
from util.file import get_files_recursive

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def read_from_fixture(path: str) -> bytes:
    with open(os.path.join(FIXTURES_DIR, path), "rb") as file:
        return file.read()


@contextmanager
def assert_read_only_queries():
    with CaptureQueriesContext(connection) as ctx:
        yield
        for query in ctx.captured_queries:
            assert query["sql"].startswith("SELECT")


@contextmanager
def assert_has_write_queries():
    with CaptureQueriesContext(connection) as ctx:
        yield
        has_writes = False
        for query in ctx.captured_queries:
            if not query["sql"].startswith("SELECT"):
                has_writes = True
                break
        assert has_writes


@pytest.fixture
def ctx():
    with tempfile.TemporaryDirectory() as temp_directory:
        yield Context(app_dir=temp_directory)


@pytest.fixture
def files(request):
    """
    Fixture to create specified files in temporary directory from dict specified as fixture args.
    The temp directory is returned as a string.
    """
    files_dict = request.param
    temp_directory = tempfile.mkdtemp()

    file_paths: list[str] = []
    for filename, content in files_dict.items():
        file_path = os.path.join(temp_directory, filename)
        file_paths.append(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            file.write(content)

    yield file_paths

    # Teardown: remove temporary directory and its contents
    import shutil

    shutil.rmtree(temp_directory)


@pytest.mark.django_db
def test_db_access(django_assert_num_queries):
    all_files = list(File.objects.all())
    assert len(all_files) == 0

    # Create a new file
    with django_assert_num_queries(1):
        file = File(
            path="test",
            size=1,
            mtime=1,
            ctime=1,
            hash="test",
            hash_method="test",
        )
        file.save()

    all_files = list(File.objects.all())
    assert len(all_files) == 1
    assert all_files[0].path == "test"
    assert all_files[0].size == 1
    assert all_files[0].mtime == 1
    assert all_files[0].ctime == 1
    assert all_files[0].hash == "test"
    assert all_files[0].hash_method == "test"


@pytest.mark.django_db
@pytest.mark.parametrize("files", [{"test": b"test"}], indirect=True)
def test_update_files_in_db(ctx, files):
    # Nothing in db
    all_files = list(File.objects.all())
    assert len(all_files) == 0

    # Update files in db
    with assert_has_write_queries():
        update_files_in_db(ctx, files, max_phase=Phase.FILE)

    # 1 file in db
    all_files = list(File.objects.all())
    assert len(all_files) == 1
    assert all_files[0].path == files[0]
    assert all_files[0].size == 4
    file = all_files[0]

    # Re-update files in db and assert nothing new is created in the db
    with assert_read_only_queries():
        update_files_in_db(ctx, files, max_phase=Phase.FILE)
    all_files = list(File.objects.all())
    assert len(all_files) == 1
    assert all_files[0] == file

    # Change the file content and assert that the file is updated in the db
    with open(files[0], "w") as f:
        f.write("test2")
    with assert_has_write_queries():
        update_files_in_db(ctx, files, max_phase=Phase.FILE)
    all_files = list(File.objects.all())
    assert len(all_files) == 1
    assert all_files[0].path == files[0]
    assert all_files[0].size == 5
    assert all_files[0].hash != file.hash


@pytest.mark.django_db
@pytest.mark.parametrize(
    "files", [{"hello.pdf": read_from_fixture("hello.pdf")}], indirect=True
)
def test_process_pdf(ctx, files, django_assert_max_num_queries):
    assert PdfTransform.objects.first() is None

    with assert_has_write_queries():
        update_files_in_db(ctx, files, max_phase=Phase.CONTENT)

    # There should be a pdf transform in the db now
    pdf_transform = PdfTransform.objects.first()
    assert pdf_transform is not None

    # Ensure output text works
    output_txt = pdf_transform.output_txt_file
    assert output_txt.read_as_text().strip() == "Hello PDF"

    # Ensure output positions work
    output_positions = pdf_transform.output_page_position_json_file
    assert output_positions.read_as_json() == [
        {"char_index": 0, "page_height": 792, "page_width": 612}
    ]

    with assert_read_only_queries():
        update_files_in_db(ctx, files, max_phase=Phase.CONTENT)


def strip_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token.strip() != ""]


@pytest.mark.django_db
@pytest.mark.parametrize(
    "files", [{"hello.pdf": read_from_fixture("hello.pdf")}], indirect=True
)
def test_process_tokens(ctx, files):
    with assert_has_write_queries():
        update_files_in_db(ctx, files, max_phase=Phase.TOKENS)

    # Ensure output tokens work
    output_tokens_json_file = TokenTransform.objects.first().output_token_json_file
    assert strip_tokens(output_tokens_json_file.read_as_json()) == ["Hello ", "PDF"]

    with assert_read_only_queries():
        update_files_in_db(ctx, files, max_phase=Phase.TOKENS)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "files", [{"hello.pdf": read_from_fixture("hello.pdf")}], indirect=True
)
def test_process_embeddings(ctx, files):
    with assert_has_write_queries():
        update_files_in_db(ctx, files, max_phase=Phase.EMBEDDINGS)

    # Ensure output embeddings work
    embedding_transform = EmbeddingTransform.objects.first()
    offsets = embedding_transform.get_offsets()
    assert offsets == [[0, 4]]
    embeddings = embedding_transform.get_embeddings(ctx)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768

    with assert_read_only_queries():
        update_files_in_db(ctx, files, max_phase=Phase.EMBEDDINGS)
