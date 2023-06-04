import json
import sys

from django.core.management import call_command

from util.context import Context
from util.database import process_file, update_files_in_db


def main() -> None:
    # # Similarity code
    # call_command("makemigrations", "db", interactive=False, verbosity=0)
    # call_command("migrate", interactive=False, verbosity=0)
    # ctx = Context()
    # # doc_1 = process_file(ctx, "path_to_doc_1.txt")
    # # doc_2 = process_file(ctx, "path_to_doc_2.txt")

    # # Pairwise similarity of all embeddings
    # similarities = []
    # for i in range(len(doc_1["embeddings"])):
    #     for j in range(len(doc_2["embeddings"])):
    #         cos_similarity = np.dot(doc_1["embeddings"][i], doc_2["embeddings"][j]) / (
    #             np.linalg.norm(doc_1["embeddings"][i])
    #             * np.linalg.norm(doc_2["embeddings"][j])
    #         )
    #         similarities.append((float(cos_similarity), i, j))

    # with open("similarities.json", "w") as f:
    #     f.write(
    #         json.dumps(
    #             {
    #                 "doc1": {
    #                     "tokens": doc_1["tokens"],
    #                     "offsets": doc_1["offsets"],
    #                 },
    #                 "doc2": {
    #                     "tokens": doc_2["tokens"],
    #                     "offsets": doc_2["offsets"],
    #                 },
    #                 "similarities": similarities,
    #             }
    #         )
    #     )

    # similarities.sort(key=lambda x: x[0], reverse=True)
    # for a, (cos_similarity, i, j) in enumerate(similarities[:100]):
    #     print(f"---------\n#{a + 1}")
    #     print(cos_similarity, i, j)
    #     print(
    #         "> "
    #         + "".join(doc_1["tokens"][doc_1["offsets"][i][0] : doc_1["offsets"][i][1]])
    #     )
    #     print(
    #         "\n> "
    #         + "".join(doc_2["tokens"][doc_2["offsets"][j][0] : doc_2["offsets"][j][1]])
    #     )

    print("MAIN")
    call_command("makemigrations", "db", interactive=False, verbosity=0)  # --merge
    call_command("migrate", interactive=False, verbosity=0)
    print("POST_MIGRATE")

    input_paths = sys.argv[1:]
    ctx = Context()
    print("UPDATING FILES IN DB")
    filenames = update_files_in_db(ctx, input_paths)

    print("NUM FILES", len(filenames))


if __name__ == "__main__":
    main()
