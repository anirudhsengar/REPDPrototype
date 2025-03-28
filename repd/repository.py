#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repository Interface Module for REPD Model

This module provides a unified interface for accessing repositories,
whether they are local filesystem directories or remote Git repositories.
It handles operations like retrieving files, commit history, and metadata.

Author: anirudhsengar
"""

import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    logger.warning("GitPython not installed. Git repository support will be limited.")
    GIT_AVAILABLE = False


class Commit:
    """Represents a commit in version control history."""

    def __init__(
        self,
        hash: str,
        author: str,
        date: datetime,
        message: str,
        modified_files: List[str],
    ):
        """
        Initialize a commit.

        Args:
            hash: Commit hash/ID
            author: Author of the commit
            date: Date of the commit
            message: Commit message
            modified_files: List of files modified in the commit
        """
        self.hash = hash
        self.author = author
        self.date = date
        self.message = message
        self.modified_files = modified_files

    def __str__(self) -> str:
        """Return string representation of the commit."""
        date_str = self.date.strftime("%Y-%m-%d %H:%M:%S")
        return f"Commit {self.hash[:7]} by {self.author} on {date_str}: {self.message[:50]}"

    def __repr__(self) -> str:
        """Return programmer representation of the commit."""
        return (
            f"Commit(hash={self.hash[:7]}, author='{self.author}', date='{self.date}')"
        )


class Repository:
    """
    Abstract base class representing a code repository.

    Provides a unified interface for accessing files and commit history
    regardless of the underlying storage mechanism.
    """

    def __init__(self, name: str):
        """
        Initialize the repository.

        Args:
            name: Repository name
        """
        self.name = name
        self.path = None

    def get_name(self) -> str:
        """
        Get repository name.

        Returns:
            Repository name
        """
        return self.name

    def get_all_files(self) -> List[str]:
        """
        Get all files in the repository.

        Returns:
            List of file paths
        """
        raise NotImplementedError("Subclasses must implement get_all_files")

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the repository.

        Args:
            file_path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement file_exists")

    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get the content of a file.

        Args:
            file_path: Path to the file

        Returns:
            File content as a string, or None if the file doesn't exist
        """
        raise NotImplementedError("Subclasses must implement get_file_content")

    def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes, or 0 if the file doesn't exist
        """
        raise NotImplementedError("Subclasses must implement get_file_size")

    def get_file_creation_date(self, file_path: str) -> datetime:
        """
        Get the creation date of a file.

        Args:
            file_path: Path to the file

        Returns:
            File creation date

        Raises:
            Exception: If the file doesn't exist
        """
        raise NotImplementedError("Subclasses must implement get_file_creation_date")

    def get_commit_history(
        self, days: int = None, author: str = None, file_path: str = None
    ) -> List[Commit]:
        """
        Get the commit history of the repository.

        Args:
            days: Number of days to include (None for all history)
            author: Filter by commit author (None for all authors)
            file_path: Filter by file path (None for all files)

        Returns:
            List of Commit objects
        """
        raise NotImplementedError("Subclasses must implement get_commit_history")

    def list_directory(self, directory_path: str) -> List[str]:
        """
        List files and directories in a directory.

        Args:
            directory_path: Path to the directory

        Returns:
            List of file and directory names
        """
        raise NotImplementedError("Subclasses must implement list_directory")

    def get_file_attributes(self, file_path: str) -> Dict[str, Any]:
        """
        Get attributes of a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary of file attributes

        Raises:
            Exception: If the file doesn't exist
        """
        raise NotImplementedError("Subclasses must implement get_file_attributes")

    def is_code_file(self, file_path: str) -> bool:
        """
        Check if a file is a code file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a code file, False otherwise
        """
        # Common code file extensions
        code_extensions = {
            # Python
            ".py",
            ".pyx",
            ".pyi",
            # JavaScript
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            # Web
            ".html",
            ".htm",
            ".css",
            ".scss",
            ".sass",
            ".less",
            # Java
            ".java",
            ".kt",
            ".groovy",
            # C/C++
            ".c",
            ".cpp",
            ".cc",
            ".h",
            ".hpp",
            ".cxx",
            # C#
            ".cs",
            # Go
            ".go",
            # Ruby
            ".rb",
            # PHP
            ".php",
            # Shell
            ".sh",
            ".bash",
            # Swift
            ".swift",
            # Rust
            ".rs",
            # Other
            ".pl",
            ".pm",
            ".scala",
            ".clj",
            ".lua",
            ".hs",
        }

        # Get file extension
        _, ext = os.path.splitext(file_path.lower())

        return ext in code_extensions

    def normalize_path(self, file_path: str) -> str:
        """
        Normalize a file path for consistent handling.

        Args:
            file_path: Path to normalize

        Returns:
            Normalized path
        """
        # Replace backslashes with forward slashes
        normalized = file_path.replace("\\", "/")

        # Remove leading slashes
        while normalized.startswith("/"):
            normalized = normalized[1:]

        # Remove drive letter for Windows paths
        if re.match(r"^[a-zA-Z]:", normalized):
            normalized = normalized[2:]

        # Remove leading ./ and ../
        normalized = re.sub(r"^(?:\./|\.\.\/)+", "", normalized)

        return normalized

    def calculate_complexity(self, content: str) -> float:
        """
        Calculate code complexity from file content.

        Args:
            content: File content

        Returns:
            Complexity score
        """
        if not content:
            return 0.0

        # Simple cyclomatic complexity estimate
        # Count branch points: if, else, for, while, etc.
        branch_keywords = [
            r"\bif\b",
            r"\belse\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\b(?:&&|\|\|)\b",
            r"\?",
            r"\bswitch\b",
        ]

        # Count matches for each keyword
        complexity = 1  # Base complexity is 1

        for keyword in branch_keywords:
            matches = re.findall(keyword, content)
            complexity += len(matches)

        # Calculate nesting depth
        lines = content.split("\n")
        max_indent = 0
        for line in lines:
            # Count leading spaces/tabs and divide by typical indentation (4)
            indent = len(line) - len(line.lstrip())
            indent_level = indent / 4
            max_indent = max(max_indent, indent_level)

        # Factor in nesting depth
        complexity += max_indent * 0.5

        # Normalize complexity score to a reasonable range
        normalized = min(1.0, complexity / 50)

        return normalized


class LocalRepository(Repository):
    """Repository implementation for local filesystems."""

    def __init__(self, path: Union[str, Path]):
        """
        Initialize the repository.

        Args:
            path: Path to the local repository
        """
        # Convert path to Path object
        path_obj = Path(path)

        # Call parent constructor with name
        super().__init__(path_obj.name)

        # Set the path after super().__init__ to avoid being overwritten
        self.path = path_obj

        logger.info(f"Initialized local repository at {self.path}")

    def get_all_files(self) -> List[str]:
        """
        Get all files in the repository.

        Returns:
            List of file paths
        """
        all_files = []

        for root, _, files in os.walk(self.path):
            for file in files:
                # Get full path
                full_path = os.path.join(root, file)

                # Convert to relative path
                rel_path = os.path.relpath(full_path, self.path)

                # Normalize path
                normalized = self.normalize_path(rel_path)

                all_files.append(normalized)

        return all_files

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the repository.

        Args:
            file_path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Check if file exists
        full_path = self.path / normalized
        return full_path.is_file()

    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get the content of a file.

        Args:
            file_path: Path to the file

        Returns:
            File content as a string, or None if the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            return None

        # Try to read file content
        try:
            # Check if file is binary
            if self._is_binary_file(full_path):
                return None

            # Read file content
            return full_path.read_text(errors="replace")

        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {str(e)}")
            return None

    def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes, or 0 if the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            return 0

        # Get file size
        return full_path.stat().st_size

    def get_file_creation_date(self, file_path: str) -> datetime:
        """
        Get the creation date of a file.

        Args:
            file_path: Path to the file

        Returns:
            File creation date

        Raises:
            Exception: If the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file stats
        stats = full_path.stat()

        # Use ctime as creation date (best approximation available)
        return datetime.fromtimestamp(stats.st_ctime)

    def get_commit_history(
        self, days: int = None, author: str = None, file_path: str = None
    ) -> List[Commit]:
        """
        Get the commit history of the repository.

        Note: For local repositories without Git, this returns an empty list.
        To get commit history, use GitRepository instead.

        Args:
            days: Number of days to include (None for all history)
            author: Filter by commit author (None for all authors)
            file_path: Filter by file path (None for all files)

        Returns:
            List of Commit objects
        """
        # Local repositories don't have commit history without Git
        logger.warning(
            "Local repositories don't have commit history. Use GitRepository instead."
        )
        return []

    def list_directory(self, directory_path: str) -> List[str]:
        """
        List files and directories in a directory.

        Args:
            directory_path: Path to the directory

        Returns:
            List of file and directory names
        """
        # Normalize path
        normalized = self.normalize_path(directory_path)

        # Get full path
        full_path = self.path / normalized

        # Check if directory exists
        if not full_path.is_dir():
            return []

        # List files and directories
        return [item.name for item in full_path.iterdir()]

    def get_file_attributes(self, file_path: str) -> Dict[str, Any]:
        """
        Get attributes of a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary of file attributes

        Raises:
            Exception: If the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file stats
        stats = full_path.stat()

        # Determine if file is binary
        is_binary = self._is_binary_file(full_path)

        return {
            "name": os.path.basename(file_path),
            "size": stats.st_size,
            "creation_date": datetime.fromtimestamp(stats.st_ctime),
            "modification_date": datetime.fromtimestamp(stats.st_mtime),
            "is_binary": is_binary,
        }

    def _is_binary_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is binary.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is binary, False otherwise
        """
        try:
            # Read first 1024 bytes
            with open(file_path, "rb") as f:
                chunk = f.read(1024)

            # Check for null bytes (common in binary files)
            if b"\0" in chunk:
                return True

            # Try to decode as UTF-8
            chunk.decode("utf-8")
            return False

        except UnicodeDecodeError:
            # If decoding fails, it's probably binary
            return True
        except:
            # If any other error occurs, assume binary to be safe
            return True


class GitRepository(Repository):
    """Repository implementation for Git repositories."""

    def __init__(self, url: str, clone_path: str = None):
        """
        Initialize the repository.

        Args:
            url: URL of the Git repository
            clone_path: Path to clone the repository to (None for temporary directory)
        """
        if not GIT_AVAILABLE:
            raise ImportError("GitPython not installed. Cannot use GitRepository.")

        self.url = url

        # Extract repository name from URL
        self.name = os.path.splitext(os.path.basename(url))[0]

        # Clone the repository
        if clone_path:
            self.path = Path(clone_path)
            self._clone_repository(self.path)
        else:
            # Create temporary directory
            self.temp_dir = tempfile.TemporaryDirectory()
            self.path = Path(self.temp_dir.name)
            self._clone_repository(self.path)

        logger.info(f"Initialized Git repository from {url} at {self.path}")

    def _clone_repository(self, path: Path) -> None:
        """
        Clone the Git repository.

        Args:
            path: Path to clone the repository to
        """
        logger.info(f"Cloning repository {self.url} to {path}")

        try:
            # Clone repository
            self.git_repo = git.Repo.clone_from(self.url, path)

        except git.GitCommandError as e:
            logger.error(f"Error cloning repository: {str(e)}")
            raise

    def get_all_files(self) -> List[str]:
        """
        Get all files in the repository.

        Returns:
            List of file paths
        """
        all_files = []

        for root, _, files in os.walk(self.path):
            for file in files:
                # Skip Git internal files
                if ".git" in root:
                    continue

                # Get full path
                full_path = os.path.join(root, file)

                # Convert to relative path
                rel_path = os.path.relpath(full_path, self.path)

                # Normalize path
                normalized = self.normalize_path(rel_path)

                all_files.append(normalized)

        return all_files

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the repository.

        Args:
            file_path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Check if file exists
        full_path = self.path / normalized
        return full_path.is_file()

    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get the content of a file.

        Args:
            file_path: Path to the file

        Returns:
            File content as a string, or None if the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            return None

        # Try to read file content
        try:
            # Check if file is binary
            if self._is_binary_file(full_path):
                return None

            # Read file content
            return full_path.read_text(errors="replace")

        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {str(e)}")
            return None

    def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes, or 0 if the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            return 0

        # Get file size
        return full_path.stat().st_size

    def get_file_creation_date(self, file_path: str) -> datetime:
        """
        Get the creation date of a file.

        For Git repositories, this returns the date of the first commit
        that introduced the file.

        Args:
            file_path: Path to the file

        Returns:
            File creation date

        Raises:
            Exception: If the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Get file history
            history = self.get_file_history(normalized)

            if history:
                # Return date of the earliest commit
                return history[-1].date

            # If no history is found, fall back to file stats
            stats = full_path.stat()
            return datetime.fromtimestamp(stats.st_ctime)

        except Exception as e:
            logger.warning(f"Error getting creation date for {file_path}: {str(e)}")

            # Fall back to file stats
            stats = full_path.stat()
            return datetime.fromtimestamp(stats.st_ctime)

    def get_commit_history(
        self, days: int = None, author: str = None, file_path: str = None
    ) -> List[Commit]:
        """
        Get the commit history of the repository.

        Args:
            days: Number of days to include (None for all history)
            author: Filter by commit author (None for all authors)
            file_path: Filter by file path (None for all files)

        Returns:
            List of Commit objects
        """
        try:
            # Get cutoff date if days specified
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
            else:
                cutoff_date = None

            # Get all commits
            git_commits = list(self.git_repo.iter_commits())

            # Convert to Commit objects
            commits = []

            for git_commit in git_commits:
                # Extract commit date
                commit_date = datetime.fromtimestamp(git_commit.committed_date)

                # Apply date filter
                if cutoff_date and commit_date < cutoff_date:
                    continue

                # Apply author filter
                if author and git_commit.author.name != author:
                    continue

                # Extract modified files
                modified_files = list(git_commit.stats.files.keys())

                # Apply file filter
                if file_path and file_path not in modified_files:
                    continue

                # Create Commit object
                commit = Commit(
                    hash=git_commit.hexsha,
                    author=git_commit.author.name,
                    date=commit_date,
                    message=git_commit.message,
                    modified_files=modified_files,
                )

                commits.append(commit)

            return commits

        except Exception as e:
            logger.error(f"Error getting commit history: {str(e)}")
            return []

    def list_directory(self, directory_path: str) -> List[str]:
        """
        List files and directories in a directory.

        Args:
            directory_path: Path to the directory

        Returns:
            List of file and directory names
        """
        # Normalize path
        normalized = self.normalize_path(directory_path)

        # Get full path
        full_path = self.path / normalized

        # Check if directory exists
        if not full_path.is_dir():
            return []

        # List files and directories
        return [item.name for item in full_path.iterdir()]

    def get_file_attributes(self, file_path: str) -> Dict[str, Any]:
        """
        Get attributes of a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary of file attributes

        Raises:
            Exception: If the file doesn't exist
        """
        # Normalize path
        normalized = self.normalize_path(file_path)

        # Get full path
        full_path = self.path / normalized

        # Check if file exists
        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file stats
        stats = full_path.stat()

        # Determine if file is binary
        is_binary = self._is_binary_file(full_path)

        # Get creation date
        try:
            creation_date = self.get_file_creation_date(file_path)
        except:
            # Fall back to file stats
            creation_date = datetime.fromtimestamp(stats.st_ctime)

        return {
            "name": os.path.basename(file_path),
            "size": stats.st_size,
            "creation_date": creation_date,
            "modification_date": datetime.fromtimestamp(stats.st_mtime),
            "is_binary": is_binary,
            "commit_count": len(self.get_file_history(file_path)),
        }

    def get_contributors(self) -> List[str]:
        """
        Get contributors to the repository.

        Returns:
            List of contributor names
        """
        try:
            # Get all commits
            git_commits = list(self.git_repo.iter_commits())

            # Extract unique authors
            contributors = set()

            for git_commit in git_commits:
                contributors.add(git_commit.author.name)

            return sorted(contributors)

        except Exception as e:
            logger.error(f"Error getting contributors: {str(e)}")
            return []

    def get_file_history(self, file_path: str) -> List[Commit]:
        """
        Get the commit history for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of Commit objects
        """
        return self.get_commit_history(file_path=file_path)

    def get_commit_stats(self, days: int = None) -> Dict[str, int]:
        """
        Get statistics about commits.

        Args:
            days: Number of days to include (None for all history)

        Returns:
            Dictionary with statistics
        """
        try:
            # Get commit history
            commits = self.get_commit_history(days=days)

            # Count total files changed
            all_changed_files = set()
            total_insertions = 0
            total_deletions = 0

            for commit in commits:
                all_changed_files.update(commit.modified_files)

                # Try to get detailed stats (if available)
                try:
                    git_commit = self.git_repo.commit(commit.hash)
                    for _, stats in git_commit.stats.files.items():
                        total_insertions += stats.get("insertions", 0)
                        total_deletions += stats.get("deletions", 0)
                except:
                    pass

            return {
                "total_commits": len(commits),
                "total_files_changed": len(all_changed_files),
                "total_insertions": total_insertions,
                "total_deletions": total_deletions,
                "total_contributors": len(self.get_contributors()),
            }

        except Exception as e:
            logger.error(f"Error getting commit stats: {str(e)}")
            return {
                "total_commits": 0,
                "total_files_changed": 0,
                "total_insertions": 0,
                "total_deletions": 0,
                "total_contributors": 0,
            }

    def _is_binary_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file is binary.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is binary, False otherwise
        """
        try:
            # Read first 1024 bytes
            with open(file_path, "rb") as f:
                chunk = f.read(1024)

            # Check for null bytes (common in binary files)
            if b"\0" in chunk:
                return True

            # Try to decode as UTF-8
            chunk.decode("utf-8")
            return False

        except UnicodeDecodeError:
            # If decoding fails, it's probably binary
            return True
        except:
            # If any other error occurs, assume binary to be safe
            return True
