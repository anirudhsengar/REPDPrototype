#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Repository Interface in REPD Model

This module contains tests for the repository abstraction layer,
which provides unified access to local and remote repositories
and their file contents, commit history, and structure.

Author: anirudhsengar
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repd.repository import Repository, GitRepository, LocalRepository, Commit


class TestRepository(unittest.TestCase):
    """Test cases for the base Repository class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a minimal abstract repository instance
        self.repo = Repository("test-repo")

    def test_init(self):
        """Test repository initialization."""
        self.assertEqual(self.repo.name, "test-repo")
        self.assertEqual(self.repo.path, None)

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        abstract_methods = [
            (self.repo.get_all_files, []),
            (self.repo.file_exists, ["file.txt"]),
            (self.repo.get_file_content, ["file.txt"]),
            (self.repo.get_file_size, ["file.txt"]),
            (self.repo.get_file_creation_date, ["file.txt"]),
            (self.repo.get_commit_history, []),
            (self.repo.list_directory, ["dir"]),
            (self.repo.get_file_attributes, ["file.txt"]),
        ]

        for method, args in abstract_methods:
            with self.assertRaises(NotImplementedError):
                method(*args)

    def test_is_code_file(self):
        """Test code file identification."""
        # Code files
        code_files = [
            "main.py",
            "app.js",
            "index.html",
            "styles.css",
            "library.cpp",
            "header.h",
            "program.java",
            "module.go",
            "script.rb",
            "app.php",
        ]

        # Non-code files
        non_code_files = [
            "image.png",
            "document.pdf",
            "data.json",  # JSON is considered data, not code
            "config.xml",  # XML is considered data, not code
            "readme.md",
            "license.txt",
            "archive.zip",
            ".gitignore",
        ]

        # Test code files
        for file in code_files:
            self.assertTrue(
                self.repo.is_code_file(file), f"Failed to identify {file} as code file"
            )

        # Test non-code files
        for file in non_code_files:
            self.assertFalse(
                self.repo.is_code_file(file),
                f"Incorrectly identified {file} as code file",
            )

    def test_normalize_path(self):
        """Test path normalization."""
        test_cases = [
            # (input, expected output)
            ("path/to/file.txt", "path/to/file.txt"),
            ("path\\to\\file.txt", "path/to/file.txt"),
            ("/path/to/file.txt", "path/to/file.txt"),
            ("./path/to/file.txt", "path/to/file.txt"),
            ("../path/to/file.txt", "path/to/file.txt"),
            ("C:\\Users\\path\\to\\file.txt", "path/to/file.txt"),
        ]

        for input_path, expected in test_cases:
            self.assertEqual(self.repo.normalize_path(input_path), expected)

    def test_calculate_complexity(self):
        """Test code complexity calculation."""
        # Test with simple code
        simple_code = "def hello():\n    print('Hello')\n    return True"
        simple_complexity = self.repo.calculate_complexity(simple_code)

        # Test with more complex code
        complex_code = """
def complex_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    else:
        if y > 0:
            return -x + y
        else:
            return -x - y

def another_function():
    for i in range(10):
        if i % 2 == 0:
            print(i)
        else:
            continue
"""
        complex_complexity = self.repo.calculate_complexity(complex_code)

        # Complex code should have higher complexity
        self.assertGreater(complex_complexity, simple_complexity)


class TestLocalRepository(unittest.TestCase):
    """Test cases for the LocalRepository class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)

        # Create some test files and directories
        sample_files = {
            "main.py": "print('Hello World')",
            "lib/utils.py": "def util_func():\n    pass",
            "lib/data.json": '{"key": "value"}',
            "docs/readme.md": "# Test Repository",
            "static/image.png": b"\x89PNG\r\n\x1a\n",  # Minimal PNG header
        }

        # Create files with content
        for file_path, content in sample_files.items():
            full_path = self.repo_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write binary or text content as appropriate
            if isinstance(content, bytes):
                full_path.write_bytes(content)
            else:
                full_path.write_text(content)

        # Create the repository object
        self.repo = LocalRepository(self.repo_path)

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test local repository initialization."""
        self.assertEqual(self.repo.name, self.repo_path.name)
        self.assertEqual(self.repo.path, self.repo_path)

    def test_get_all_files(self):
        """Test retrieving all files from the repository."""
        files = self.repo.get_all_files()

        # Check that all expected files are listed
        expected_files = [
            "main.py",
            "lib/utils.py",
            "lib/data.json",
            "docs/readme.md",
            "static/image.png",
        ]

        # Convert to sets for comparison (order doesn't matter)
        self.assertEqual(set(files), set(expected_files))

    def test_file_exists(self):
        """Test file existence checking."""
        # Test existing files
        self.assertTrue(self.repo.file_exists("main.py"))
        self.assertTrue(self.repo.file_exists("lib/utils.py"))

        # Test non-existent files
        self.assertFalse(self.repo.file_exists("nonexistent.py"))
        self.assertFalse(self.repo.file_exists("lib/nonexistent.py"))

    def test_get_file_content(self):
        """Test file content retrieval."""
        # Test text files
        self.assertEqual(self.repo.get_file_content("main.py"), "print('Hello World')")

        self.assertEqual(
            self.repo.get_file_content("lib/utils.py"), "def util_func():\n    pass"
        )

        # Test binary files (should return None or empty for binary files)
        self.assertIn(
            self.repo.get_file_content("static/image.png"),
            [None, "", b"\x89PNG\r\n\x1a\n"],
        )

        # Test non-existent file
        self.assertIsNone(self.repo.get_file_content("nonexistent.py"))

    def test_get_file_size(self):
        """Test file size retrieval."""
        self.assertEqual(self.repo.get_file_size("main.py"), 19)
        self.assertEqual(self.repo.get_file_size("lib/utils.py"), 23)

        # Test non-existent file
        self.assertEqual(self.repo.get_file_size("nonexistent.py"), 0)

    def test_get_file_creation_date(self):
        """Test file creation date retrieval."""
        # Should return a datetime object for existing files
        self.assertIsInstance(self.repo.get_file_creation_date("main.py"), datetime)

        # Test non-existent file
        with self.assertRaises(Exception):
            self.repo.get_file_creation_date("nonexistent.py")

    def test_list_directory(self):
        """Test directory listing."""
        # Test root directory
        root_files = self.repo.list_directory("")
        self.assertIn("main.py", root_files)

        # Test subdirectory
        lib_files = self.repo.list_directory("lib")
        self.assertIn("utils.py", lib_files)
        self.assertIn("data.json", lib_files)

        # Test non-existent directory
        self.assertEqual(self.repo.list_directory("nonexistent"), [])

    def test_get_file_attributes(self):
        """Test retrieving file attributes."""
        attrs = self.repo.get_file_attributes("main.py")

        # Check returned attributes
        self.assertIn("size", attrs)
        self.assertIn("creation_date", attrs)
        self.assertIn("is_binary", attrs)

        # Check values
        self.assertEqual(attrs["size"], 19)
        self.assertFalse(attrs["is_binary"])

        # Test non-existent file
        with self.assertRaises(Exception):
            self.repo.get_file_attributes("nonexistent.py")


class TestGitRepository(unittest.TestCase):
    """Test cases for the GitRepository class."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        try:
            import git

            cls.git_available = True
        except ImportError:
            cls.git_available = False

    def setUp(self):
        """Set up test environment before each test."""
        if not self.git_available:
            self.skipTest("GitPython package not available")

        # Create a temporary directory for the git repo
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)

        # Initialize a git repository and create test files
        with patch("repd.repository.git.Repo.clone_from") as mock_clone:
            # Mock the clone operation to create a fake Git repo
            mock_clone.return_value = MagicMock()

            # Create a GitRepository instance with a mocked Git backend
            self.repo = GitRepository("https://github.com/example/test-repo.git")
            self.repo.path = self.repo_path  # Set the repo path manually

            # Mock the git.Repo instance
            self.repo.git_repo = MagicMock()

            # Create some test files to work with
            sample_files = {
                "main.py": "print('Hello World')",
                "lib/utils.py": "def util_func():\n    pass",
                "docs/readme.md": "# Test Repository",
            }

            # Create files with content
            for file_path, content in sample_files.items():
                full_path = self.repo_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Mock the git_repo.iter_commits method
            mock_commits = []
            base_date = datetime.now() - timedelta(days=10)

            # Mock commit objects
            for i in range(5):
                mock_commit = MagicMock()
                mock_commit.hexsha = f"commit{i}"
                mock_commit.author.name = f"Author{i % 2}"
                mock_commit.committed_datetime = base_date + timedelta(days=i)
                mock_commit.message = f"Commit message {i}"

                # Mock the file changes for this commit
                if i == 0:
                    changed_files = ["main.py", "docs/readme.md"]
                elif i == 1:
                    changed_files = ["lib/utils.py"]
                elif i == 2:
                    changed_files = ["main.py"]
                else:
                    changed_files = ["docs/readme.md"]

                mock_commit.stats.files = {
                    file: {"insertions": 5, "deletions": 2} for file in changed_files
                }
                mock_commits.append(mock_commit)

            # Set the mocked commits
            self.repo.git_repo.iter_commits.return_value = mock_commits

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test git repository initialization."""
        # Create a new instance that exercises the __init__ method
        with patch("repd.repository.git.Repo.clone_from") as mock_clone:
            mock_clone.return_value = MagicMock()
            repo = GitRepository("https://github.com/user/repo.git")

            self.assertEqual(repo.url, "https://github.com/user/repo.git")
            self.assertEqual(repo.name, "repo")

    def test_get_commit_history(self):
        """Test retrieving commit history."""
        commits = self.repo.get_commit_history()

        # Check that we have the right number of commits
        self.assertEqual(len(commits), 5)

        # Check that each commit has the right properties
        for commit in commits:
            self.assertIsInstance(commit, Commit)
            self.assertTrue(hasattr(commit, "hash"))
            self.assertTrue(hasattr(commit, "author"))
            self.assertTrue(hasattr(commit, "date"))
            self.assertTrue(hasattr(commit, "message"))
            self.assertTrue(hasattr(commit, "modified_files"))

    def test_get_commit_history_with_filters(self):
        """Test retrieving commit history with filters."""
        # Test filtering by days
        recent_commits = self.repo.get_commit_history(days=3)
        self.assertLessEqual(len(recent_commits), 3)

        # Test filtering by author
        author_commits = self.repo.get_commit_history(author="Author0")
        self.assertGreaterEqual(len(author_commits), 1)
        for commit in author_commits:
            self.assertEqual(commit.author, "Author0")

        # Test filtering by file
        file_commits = self.repo.get_commit_history(file_path="main.py")
        self.assertGreaterEqual(len(file_commits), 2)

        # Test combination of filters
        filtered_commits = self.repo.get_commit_history(
            days=5, author="Author1", file_path="lib/utils.py"
        )

        # Verify that filtered commits match all criteria
        for commit in filtered_commits:
            self.assertEqual(commit.author, "Author1")
            self.assertIn("lib/utils.py", commit.modified_files)

    def test_get_contributors(self):
        """Test retrieving repository contributors."""
        contributors = self.repo.get_contributors()

        # Should have at least the two authors we defined
        self.assertGreaterEqual(len(contributors), 2)
        self.assertIn("Author0", contributors)
        self.assertIn("Author1", contributors)

    def test_get_file_history(self):
        """Test retrieving history for a specific file."""
        # Get history for main.py
        file_history = self.repo.get_file_history("main.py")

        # Should have commits where main.py was modified
        self.assertGreaterEqual(len(file_history), 2)

        # Each entry should refer to main.py
        for commit in file_history:
            self.assertIn("main.py", commit.modified_files)

    def test_get_commit_stats(self):
        """Test retrieving commit statistics."""
        # Get overall stats
        stats = self.repo.get_commit_stats()

        # Should include basic statistics
        self.assertIn("total_commits", stats)
        self.assertIn("total_files_changed", stats)
        self.assertIn("total_insertions", stats)
        self.assertIn("total_deletions", stats)

        # Check specific values
        self.assertEqual(stats["total_commits"], 5)

        # Get stats for a specific time period
        recent_stats = self.repo.get_commit_stats(days=3)
        self.assertLessEqual(recent_stats["total_commits"], 3)


class TestCommit(unittest.TestCase):
    """Test cases for the Commit class."""

    def test_init(self):
        """Test commit initialization."""
        # Create a commit
        date = datetime.now()
        commit = Commit(
            hash="abc123",
            author="test_author",
            date=date,
            message="Test commit",
            modified_files=["file1.py", "file2.py"],
        )

        # Check properties
        self.assertEqual(commit.hash, "abc123")
        self.assertEqual(commit.author, "test_author")
        self.assertEqual(commit.date, date)
        self.assertEqual(commit.message, "Test commit")
        self.assertEqual(commit.modified_files, ["file1.py", "file2.py"])

    def test_str(self):
        """Test string representation."""
        date = datetime(2025, 3, 26, 7, 11, 48)
        commit = Commit(
            hash="abc123",
            author="test_author",
            date=date,
            message="Test commit",
            modified_files=["file1.py", "file2.py"],
        )

        # Check string representation
        commit_str = str(commit)
        self.assertIn("abc123", commit_str)
        self.assertIn("test_author", commit_str)
        self.assertIn("2025-03-26", commit_str)

    def test_repr(self):
        """Test repr representation."""
        date = datetime(2025, 3, 26, 7, 11, 48)
        commit = Commit(
            hash="abc123",
            author="test_author",
            date=date,
            message="Test commit",
            modified_files=["file1.py", "file2.py"],
        )

        # Check repr representation
        commit_repr = repr(commit)
        self.assertIn("Commit", commit_repr)
        self.assertIn("abc123", commit_repr)


if __name__ == "__main__":
    unittest.main()
