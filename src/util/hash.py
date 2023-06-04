import hashlib


def generate_file_sha256(absolute_filename, blocksize=2**20):
    # https://stackoverflow.com/a/1131255
    m = hashlib.sha256()
    with open(absolute_filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()
