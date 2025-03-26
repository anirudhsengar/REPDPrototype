#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repository Interface for REPD Model

This module provides an abstraction layer for interacting with Git repositories,
allowing the REPD model to extract commits, changes, file contents, and other
repository information.

Author: anirudhsengar
"""

import logging
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import git
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Repository:
    """
    Interface for interacting with Git repositories.

    This class provides methods for extracting repository information,
    including commits, changes, file contents, and more. It uses the
    GitPython library for most operations, with fallbacks to direct
    git commands when needed for performance.
    """

    # Extensions for common code file types
    CODE_FILE_EXTENSIONS = {
        # Python
        ".py", ".pyx", ".pyi", ".pyw",
        # JavaScript/TypeScript
        ".js", ".jsx", ".ts", ".tsx",
        # Java
        ".java", ".kt", ".scala", ".groovy",
        # C/C++
        ".c", ".cpp", ".cc", ".h", ".hpp", ".cxx",
        # Go
        ".go",
        # Ruby
        ".rb",
        # PHP
        ".php",
        # Rust
        ".rs",
        # Swift
        ".swift",
        # C#
        ".cs",
        # Shell
        ".sh", ".bash", ".zsh",
        # Configuration
        ".json", ".yaml", ".yml", ".toml", ".xml",
        # Other
        ".md", ".rst", ".txt"  # Documentation
    }

    # Files/paths to ignore
    IGNORE_PATTERNS = [
        r".git/",
        r"node_modules/",
        r"__pycache__/",
        r"\.venv/",
        r"venv/",
        r"\.pytest_cache/",
        r"\.idea/",
        r"\.vs/",
        r"\.vscode/",
        r"\.github/",
        r"dist/",
        r"build/",
        r"\.DS_Store$",
        r".*\.min\.js$",
        r".*\.min\.css$"
    ]

    def __init__(self, path: str):
        """
        Initialize a repository interface.

        Args:
            path: Path to the Git repository
        """
        self.path = os.path.abspath(path)

        # Validate repository path
        if not os.path.isdir(self.path):
            raise ValueError(f"Repository path does not exist: {self.path}")

        # Check if path is a Git repository
        if not os.path.isdir(os.path.join(self.path, ".git")):
            raise ValueError(f"Not a Git repository: {self.path}")

        # Initialize Git repository
        try:
            self.repo = git.Repo(self.path)
            logger.info(f"Initialized repository: {self.path}")

            # Get basic repository information
            remote_urls = [remote.url for remote in self.repo.remotes]
            remote_info = ", ".join(remote_urls) if remote_urls else "No remotes configured"
            logger.debug(f"Repository remotes: {remote_info}")

        except git.InvalidGitRepositoryError:
            raise ValueError(f"Invalid Git repository: {self.path}")
        except Exception as e:
            raise ValueError(f"Error initializing repository: {str(e)}")

        # Compiled ignore patterns
        self.ignore_patterns = [re.compile(pattern) for pattern in self.IGNORE_PATTERNS]

    def get_commit_history(
            self,
            limit: int = 1000,
            branch: str = "HEAD",
            skip_merges: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get commit history from the repository.

        Args:
            limit: Maximum number of commits to retrieve
            branch: Branch name or reference to retrieve commits from
            skip_merges: Whether to skip merge commits

        Returns:
            List of commit dictionaries with metadata and changed files
        """
        logger.debug(f"Retrieving commit history (limit: {limit}, branch: {branch})")

        # Get commits using GitPython
        try:
            # Create iterator for commits
            commits_iter = self.repo.iter_commits(branch, max_count=limit)

            # Process each commit
            commits = []
            for commit in tqdm(commits_iter, desc="Loading commits", total=min(limit, 1000), unit="commit"):
                # Skip merge commits if requested
                if skip_merges and len(commit.parents) > 1:
                    continue

                # Extract basic commit information
                commit_data = {
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:7],
                    "author": commit.author.email,
                    "author_name": commit.author.name,
                    "message": commit.message.strip(),
                    "timestamp": datetime.fromtimestamp(commit.committed_date),
                    "is_merge": len(commit.parents) > 1,
                    "files": []
                }

                # Get changed files using git diff-tree directly for better performance
                # This is much faster than using commit.stats for large repositories
                try:
                    if len(commit.parents) > 0:
                        parent = commit.parents[0]
                        diff_index = parent.diff(commit)

                        # Extract changed files
                        for diff in diff_index:
                            path = diff.a_path if diff.a_path else diff.b_path

                            # Skip ignored paths
                            if self._should_ignore_path(path):
                                continue

                            commit_data["files"].append(path)
                    else:
                        # For first commit, get all files
                        for item in commit.tree.traverse():
                            if item.type == "blob":  # Only include files, not directories
                                path = item.path

                                # Skip ignored paths
                                if self._should_ignore_path(path):
                                    continue

                                commit_data["files"].append(path)

                except Exception as e:
                    logger.warning(f"Error getting files for commit {commit.hexsha}: {str(e)}")

                commits.append(commit_data)

            logger.debug(f"Retrieved {len(commits)} commits")
            return commits

        except Exception as e:
            logger.error(f"Error retrieving commit history: {str(e)}")
            return []

    def get_file_at_commit(
            self,
            filename: str,
            commit_hash: str = "HEAD"
    ) -> Optional[str]:
        """
        Get file content at a specific commit.

        Args:
            filename: Path to the file relative to repository root
            commit_hash: Commit hash or reference

        Returns:
            File content as string or None if file doesn't exist
        """
        try:
            # Normalize filename to repository path format
            filename = self.normalize_path(filename)

            # Get the commit object
            commit = self.repo.commit(commit_hash)

            # Get the blob object for the file
            try:
                blob = commit.tree / filename
                return blob.data_stream.read().decode('utf-8', errors='replace')
            except KeyError:
                # File doesn't exist in this commit
                return None

        except Exception as e:
            logger.warning(f"Error getting file {filename} at {commit_hash}: {str(e)}")
            return None

    def get_file_content(self, filename: str) -> Optional[str]:
        """
        Get content of a file in the repository.

        Args:
            filename: Path to the file relative to repository root

        Returns:
            File content as string or None if file doesn't exist
        """
        # Normalize filename to repository path format
        filename = self.normalize_path(filename)

        # Construct absolute path
        abs_path = os.path.join(self.path, filename)

        # Check if file exists
        if not os.path.isfile(abs_path):
            return None

        # Read file content
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading file {filename}: {str(e)}")
            return None

    def normalize_path(self, path: str) -> str:
        """
        Normalize a path to repository format.

        Args:
            path: File path to normalize

        Returns:
            Normalized path
        """
        # Replace backslashes with forward slashes
        normalized = path.replace('\\', '/')

        # Remove leading slashes and repo directory prefix
        repo_dir = os.path.basename(self.path)
        patterns = [
            f"^/+",  # Leading slashes
            f"^{re.escape(repo_dir)}/+"  # Repository name prefix
        ]

        for pattern in patterns:
            normalized = re.sub(pattern, '', normalized)

        return normalized

    def get_file_history(
            self,
            filename: str,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get the commit history for a specific file.

        Args:
            filename: Path to the file relative to repository root
            limit: Maximum number of commits to retrieve

        Returns:
            List of commit dictionaries for the file
        """
        # Normalize filename to repository path format
        filename = self.normalize_path(filename)

        try:
            # Use git log command directly for better performance
            cmd = [
                "git", "-C", self.path,
                "log", "--follow", f"-{limit}", "--name-status",
                "--pretty=format:%H|%an|%ae|%at|%s",
                "--", filename
            ]

            output = subprocess.check_output(cmd, universal_newlines=True)

            # Parse output
            commits = []
            commit_data = None

            for line in output.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Check if this is a commit line
                if '|' in line:
                    # If we already have commit data, add it to commits
                    if commit_data:
                        commits.append(commit_data)

                    # Parse commit data
                    parts = line.split('|')
                    if len(parts) >= 5:
                        commit_hash, author_name, author_email, timestamp, message = parts[0:5]

                        commit_data = {
                            "hash": commit_hash,
                            "short_hash": commit_hash[:7],
                            "author": author_email,
                            "author_name": author_name,
                            "message": message,
                            "timestamp": datetime.fromtimestamp(int(timestamp)),
                            "file": filename,
                            "change_type": ""
                        }

                # Check if this is a file status line
                elif commit_data and line[0] in ['M', 'A', 'D', 'R']:
                    change_type = {
                        'M': 'modified',
                        'A': 'added',
                        'D': 'deleted',
                        'R': 'renamed'
                    }.get(line[0], 'unknown')

                    commit_data["change_type"] = change_type

            # Add the last commit data if exists
            if commit_data:
                commits.append(commit_data)

            return commits

        except Exception as e:
            logger.warning(f"Error getting history for file {filename}: {str(e)}")
            return []

    def get_file_blame(self, filename: str) -> List[Dict[str, Any]]:
        """
        Get blame information for a file.

        Args:
            filename: Path to the file relative to repository root

        Returns:
            List of dictionaries with line-by-line blame information
        """
        # Normalize filename to repository path format
        filename = self.normalize_path(filename)

        try:
            # Get blame information
            blame_data = []

            # Use git blame command directly
            cmd = [
                "git", "-C", self.path,
                "blame", "-p", "--", filename
            ]

            output = subprocess.check_output(cmd, universal_newlines=True)

            # Parse output
            current_commit = None
            current_line = None

            for line in output.split('\n'):
                # Skip empty lines
                if not line:
                    continue

                # Commit header line
                if line.startswith('^') or re.match(r'^[0-9a-f]{40}', line):
                    parts = line.split()
                    commit_hash = parts[0].lstrip('^')
                    line_num = int(parts[2])

                    current_commit = {
                        "hash": commit_hash,
                        "short_hash": commit_hash[:7],
                        "line_num": line_num,
                        "author": "",
                        "author_name": "",
                        "timestamp": None,
                        "line_content": ""
                    }

                # Author line
                elif line.startswith('author '):
                    if current_commit:
                        current_commit["author_name"] = line[7:]

                # Author email line
                elif line.startswith('author-mail '):
                    if current_commit:
                        current_commit["author"] = line[13:].strip('<>')

                # Author time line
                elif line.startswith('author-time '):
                    if current_commit:
                        timestamp = int(line[12:])
                        current_commit["timestamp"] = datetime.fromtimestamp(timestamp)

                # Line content
                elif line.startswith('\t'):
                    if current_commit:
                        current_commit["line_content"] = line[1:]  # Remove tab character
                        blame_data.append(current_commit.copy())

            return blame_data

        except Exception as e:
            logger.warning(f"Error getting blame for file {filename}: {str(e)}")
            return []

    def get_all_files(self) -> List[str]:
        """
        Get list of all files in the repository.

        Returns:
            List of file paths relative to repository root
        """
        all_files = []

        # Walk through repository directory
        for root, dirs, files in os.walk(self.path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore_path(os.path.join(root, d))]

            # Process files
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.path)

                # Convert to standard repository format
                rel_path = self.normalize_path(rel_path)

                # Skip ignored files
                if self._should_ignore_path(rel_path):
                    continue

                all_files.append(rel_path)

        return all_files

    def is_code_file(self, filename: str) -> bool:
        """
        Check if a file is a code file.

        Args:
            filename: Path to the file

        Returns:
            True if file is a code file, False otherwise
        """
        # Get file extension
        _, extension = os.path.splitext(filename.lower())

        # Check if extension is in code file extensions
        return extension in self.CODE_FILE_EXTENSIONS

    def file_exists(self, filename: str) -> bool:
        """
        Check if a file exists in the repository.

        Args:
            filename: Path to the file relative to repository root

        Returns:
            True if file exists, False otherwise
        """
        # Normalize filename to repository path format
        filename = self.normalize_path(filename)

        # Check if file exists
        file_path = os.path.join(self.path, filename)
        return os.path.isfile(file_path)

    def get_file_size(self, filename: str) -> int:
        """
        Get size of a file in bytes.

        Args:
            filename: Path to the file relative to repository root

        Returns:
            File size in bytes or 0 if file doesn't exist
        """
        # Normalize filename to repository path format
        filename = self.normalize_path(filename)

        # Get file size
        file_path = os.path.join(self.path, filename)
        if os.path.isfile(file_path):
            return os.path.getsize(file_path)
        else:
            return 0

    def _should_ignore_path(self, path: str) -> bool:
        """
        Check if a path should be ignored based on ignore patterns.

        Args:
            path: Path to check

        Returns:
            True if path should be ignored, False otherwise
        """
        # Normalize path for consistency
        norm_path = path.replace('\\', '/')

        # Check against ignore patterns
        for pattern in self.ignore_patterns:
            if pattern.search(norm_path):
                return True

        return False

    def get_commit_stats(self, weeks: int = 52) -> Dict[str, Any]:
        """
        Get repository statistics for a time period.

        Args:
            weeks: Number of weeks to analyze

        Returns:
            Dictionary with repository statistics
        """
        try:
            # Use git shortlog and git log commands
            shortlog_cmd = [
                "git", "-C", self.path,
                "shortlog", "-s", "-n", "--all", "--no-merges"
            ]

            activity_cmd = [
                "git", "-C", self.path,
                "log", "--all", "--format=format:%at", "--since", f"{weeks} weeks ago"
            ]

            # Get contributor statistics
            shortlog_output = subprocess.check_output(shortlog_cmd, universal_newlines=True)

            contributors = []
            total_commits = 0

            for line in shortlog_output.strip().split('\n'):
                if not line.strip():
                    continue

                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    commit_count = int(parts[0].strip())
                    author = parts[1].strip()

                    contributors.append({
                        "name": author,
                        "commits": commit_count
                    })
                    total_commits += commit_count

            # Get activity data (timestamp of each commit)
            activity_output = subprocess.check_output(activity_cmd, universal_newlines=True)
            timestamps = [int(ts) for ts in activity_output.strip().split('\n') if ts.strip()]

            # Group by week
            weekly_activity = [0] * weeks
            if timestamps:
                # Find the most recent timestamp
                max_ts = max(timestamps)

                # Calculate week number for each timestamp
                for ts in timestamps:
                    # Calculate weeks ago (0 = current week)
                    weeks_ago = int((max_ts - ts) / (7 * 24 * 3600))

                    # Increment the appropriate week bucket
                    if 0 <= weeks_ago < weeks:
                        weekly_activity[weeks_ago] += 1

            # Compile statistics
            stats = {
                "total_commits": total_commits,
                "total_contributors": len(contributors),
                "top_contributors": contributors[:10],
                "weekly_activity": list(reversed(weekly_activity)),  # Oldest to newest
                "weeks_analyzed": weeks
            }

            return stats

        except Exception as e:
            logger.warning(f"Error getting repository statistics: {str(e)}")
            return {}


if __name__ == "__main__":
    # Example usage
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python repository.py /path/to/repository")
        sys.exit(1)

    # Initialize repository
    repo_path = sys.argv[1]
    repository = Repository(repo_path)

    # Get and print repository information
    print(f"Repository: {repository.path}")

    # Get all files
    files = repository.get_all_files()
    print(f"Total files: {len(files)}")
    print("Sample files:")
    for file in files[:5]:
        print(f" - {file}")

    # Get commit history
    commits = repository.get_commit_history(limit=10)
    print("\nRecent commits:")
    for commit in commits:
        print(f" - {commit['short_hash']} | {commit['author']} | {commit['timestamp']} | {commit['message'][:50]}")

    # Get repository statistics
    stats = repository.get_commit_stats(weeks=4)
    print(f"\nTotal commits: {stats.get('total_commits', 'N/A')}")
    print(f"Total contributors: {stats.get('total_contributors', 'N/A')}")

    # Print weekly activity
    print("\nActivity (last 4 weeks, oldest to newest):")
    weekly = stats.get('weekly_activity', [])
    for i, commits in enumerate(weekly):
        print(f" - Week {i + 1}: {commits} commits")