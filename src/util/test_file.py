import os
import shutil
import tempfile
import unittest

from .file import get_files_recursive


def set_up_tmp_files_dirs(files_and_dirs: list[str]) -> None:
    for item in files_and_dirs:
        if os.path.splitext(item)[-1] != "":
            # Create intermediate directories
            os.makedirs(os.path.dirname(item), exist_ok=True)
            with open(item, "w") as f:
                f.write("test")
        else:
            os.makedirs(item, exist_ok=True)


class TestGetFilesRecursive(unittest.TestCase):
    def setUp(self) -> None:
        # Set up temporary files and directories for testing
        self.tmp_dir = tempfile.mkdtemp()
        self.files_and_dirs = [
            os.path.join(self.tmp_dir, "dir1", "file1.txt"),
            os.path.join(self.tmp_dir, "dir1", "file2.txt"),
            os.path.join(self.tmp_dir, "dir2", "dir3", "file3.txt"),
            os.path.join(self.tmp_dir, "dir2", "dir3", "file4.txt"),
            os.path.join(self.tmp_dir, "file5.txt"),
            os.path.join(self.tmp_dir, "dir3"),
        ]
        set_up_tmp_files_dirs(self.files_and_dirs)

    def tearDown(self) -> None:
        # Remove the tmp dir
        shutil.rmtree(self.tmp_dir)

    def test_get_files_recursive(self):
        # Test case 1: Single file
        single_file = [self.files_and_dirs[0]]
        expected_output = [self.files_and_dirs[0]]
        self.assertEqual(list(get_files_recursive(single_file)), expected_output)

        # Test case 2: Single directory
        single_dir = [os.path.join(self.tmp_dir, "dir1")]
        expected_output = self.files_and_dirs[:2]
        self.assertEqual(
            sorted(list(get_files_recursive(single_dir))), sorted(expected_output)
        )

        # Test case 3: Multiple files and directories
        files_and_dirs_input = [
            self.files_and_dirs[4],
            os.path.join(self.tmp_dir, "dir1"),
            os.path.join(self.tmp_dir, "dir2"),
        ]
        expected_output = self.files_and_dirs[:-1]
        self.assertEqual(
            sorted(list(get_files_recursive(files_and_dirs_input))),
            sorted(expected_output),
        )

        # Test case 4: all paths
        all_paths = [self.tmp_dir]
        expected_output = self.files_and_dirs[:-1]
        self.assertEqual(
            sorted(list(get_files_recursive(all_paths))), sorted(expected_output)
        )

        # Test case 5: Empty list
        self.assertEqual(list(get_files_recursive([])), [])

        # Test case 6: Non-existent path
        non_existent_path = [os.path.join(self.tmp_dir, "non_existent")]
        self.assertEqual(list(get_files_recursive(non_existent_path)), [])


if __name__ == "__main__":
    unittest.main()
