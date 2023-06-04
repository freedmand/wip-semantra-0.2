# An old reference file of how to speed up db operations
# There may be a way to speed up processing by intelligently incorporating
# some of these optimizations later

import asyncio
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from multiprocessing.pool import Pool, ThreadPool

from db.models import File

from .file import get_files_recursive
from .hash import generate_file_md5


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def get_existing_files_in_db(paths: list[str], batch_size: int) -> dict[str, File]:
    existing_files = {}

    for idx in range(0, len(paths), batch_size):
        batch_paths = paths[idx : idx + batch_size]
        batch_files = File.objects.filter(path__in=batch_paths)

        for file in batch_files:
            existing_files[file.path] = file

    return existing_files


def update_file_in_db(
    filename: str,
    existing_files: dict[str, File],
) -> None:
    updates: list[File] = []
    creates: list[File] = []
    try:
        stats = os.stat(filename)
    except:
        return updates, creates
    path = os.path.abspath(filename)
    if os.path.isdir(path):
        return updates, creates
    size, mtime, ctime = stats.st_size, stats.st_mtime_ns, stats.st_ctime_ns
    existing_file = existing_files.get(path)
    if existing_file:
        if (
            existing_file.size != size
            or existing_file.mtime != mtime
            or existing_file.ctime != ctime
        ):
            updates.append((existing_file, path, size, mtime, ctime))
        else:
            return updates, creates
    else:
        creates.append((path, size, mtime, ctime))

    return updates, creates


def update_file_in_db_wrapper(args: tuple) -> None:
    return update_file_in_db(args[0], args[1])


def compute_md5_create(path, size, mtime, ctime) -> None:
    hash = generate_file_md5(path)
    return File(
        path=path,
        size=size,
        mtime=mtime,
        ctime=ctime,
        hash=hash,
        hash_method="md5",
    )


def compute_md5_update(existing_file, path, size, mtime, ctime) -> None:
    hash = generate_file_md5(path)
    existing_file.update(
        size=size, mtime=mtime, ctime=ctime, hash=hash, hash_method="md5"
    )
    return existing_file


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
            hash = generate_file_md5(filename)
            existing_file.update(
                size=size, mtime=mtime, ctime=ctime, hash=hash, hash_method="md5"
            )
            return existing_file
        else:
            return existing_file
    else:
        hash = generate_file_md5(filename)
        file = File(
            path=filename,
            size=size,
            mtime=mtime,
            ctime=ctime,
            hash=hash,
            hash_method="md5",
        )
        file.save()
        return file


def update_files_in_db(filenames: list[str]) -> list[str]:
    for filename in filenames:
        get_file(filename)
    return filenames


def update_files_in_db_fast(filenames: list[str], chunk_size=950) -> list[str]:
    all_paths = []

    updates = []
    creates = []

    def update_callback(result):
        updates.append(result)
        if len(updates) > chunk_size:
            bulk_updates = updates[:]
            updates.clear()
            File.objects.bulk_update(
                bulk_updates,
                [
                    "size",
                    "mtime",
                    "ctime",
                    "hash",
                    "hash_method",
                ],
            )

    def create_callback(result):
        creates.append(result)
        if len(creates) > chunk_size:
            bulk_creates = creates[:]
            creates.clear()
            File.objects.bulk_create(bulk_creates)

    with Pool() as pool:
        with ThreadPool() as thread_pool:
            for chunk in batcher(get_files_recursive(filenames), chunk_size):
                paths: list[str] = [os.path.abspath(fn) for fn in chunk]
                all_paths += paths
                existing_files = get_existing_files_in_db(paths, chunk_size)
                input_tuples = [(fn, existing_files) for fn in chunk]

                for sub_updates, sub_creates in thread_pool.imap_unordered(
                    update_file_in_db_wrapper, input_tuples
                ):
                    for update in sub_updates:
                        pool.apply_async(
                            compute_md5_update, args=update, callback=update_callback
                        )

                    for create in sub_creates:
                        pool.apply_async(
                            compute_md5_create, args=create, callback=create_callback
                        )

        # Wait for all tasks to complete
        pool.close()
        pool.join()

    print(f"Updating {len(updates)} items")
    print(f"Creating {len(creates)} items")
    File.objects.bulk_update(
        updates,
        [
            "size",
            "mtime",
            "ctime",
            "hash",
            "hash_method",
        ],
    )
    File.objects.bulk_create(creates)

    return all_paths
