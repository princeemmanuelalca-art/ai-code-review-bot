"""
AI Code Review Bot
Fetches PR diffs, sends them to Groq API for review, and posts structured comments.
"""

import os
import re
import sys
import time
from typing import List, Dict, Any

from github import Github
from groq import Groq

# Configuration
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS_PER_CHUNK = 4000  # Conservative limit for diff chunks
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

SYSTEM_PROMPT = """You are an expert code reviewer. Analyze the provided code diff and provide a structured review covering:

1. Bugs - Logic errors, null pointer risks, off-by-one errors, incorrect API usage
2. Security Issues - Injection vulnerabilities, hardcoded secrets, unsafe deserialization, auth flaws, input validation
3. Performance - Inefficient algorithms, N+1 queries, memory leaks, unnecessary computations
4. Style - Code readability, naming conventions, PEP 8 compliance, docstring quality

Format your response EXACTLY as follows:

## Summary
Brief overview of the changes and overall assessment (2-3 sentences).

## Issues Found

### High Severity
- **[Category]**: Description of issue and location (file:line if identifiable)
  - **Impact**: Why this matters
  - **Fix**: Concrete suggestion to resolve

### Medium Severity
- **[Category]**: Description of issue and location
  - **Impact**: Why this matters
  - **Fix**: Concrete suggestion to resolve

### Low Severity
- **[Category]**: Description of issue and location
  - **Impact**: Why this matters
  - **Fix**: Concrete suggestion to resolve

## Suggestions
- General improvements not tied to specific issues (refactoring, testing, documentation)

If no issues found in a severity level, write "No issues found."
Be concise but thorough. Focus on actionable feedback."""


def get_env_var(name: str) -> str:
    """Get environment variable or exit."""
    value = os.environ.get(name)
    if not value:
        print(f"Error: {name} environment variable is required")
        sys.exit(1)
    return value


def init_clients() -> tuple[Github, Groq]:
    """Initialize GitHub and Groq clients."""
    github_token = get_env_var("GITHUB_TOKEN")
    groq_api_key = get_env_var("GROQ_API_KEY")

    github_client = Github(github_token)
    groq_client = Groq(api_key=groq_api_key)

    return github_client, groq_client


def get_pr_diff(github_client: Github, repo_name: str, pr_number: int) -> str:
    """Fetch the full PR diff."""
    repo = github_client.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Get the diff using the raw diff URL
    headers = {"Accept": "application/vnd.github.v3.diff"}
    diff_url = pr.diff_url

    # Use PyGithub's requester for raw diff
    requester = github_client._Github__requester
    status, headers, content = requester.requestJsonAndCheck("GET", diff_url)

    return content


def chunk_diff(diff_text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> List[str]:
    """
    Split diff into chunks by file, ensuring each chunk stays under token limit.
    Uses approximate token count (1 token ≈ 4 chars for code).
    """
    if not diff_text.strip():
        return []

    # Split diff by file boundaries
    file_pattern = r"(?=diff --git)"
    files = re.split(file_pattern, diff_text)
    files = [f.strip() for f in files if f.strip()]

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for file_diff in files:
        # Approximate tokens: chars / 4
        file_tokens = len(file_diff) // 4

        if file_tokens > max_tokens:
            # Single file is too large, split by hunks
            hunk_pattern = r"(?=@@ -\d+,?\d* \\+\d+,?\d* @@)"
            hunks = re.split(hunk_pattern, file_diff)
            header = hunks[0] if hunks else ""

            for hunk in hunks[1:]:
                hunk_text = header + hunk
                hunk_tokens = len(hunk_text) // 4

                if current_tokens + hunk_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = header + hunk
                    current_tokens = len(header) // 4 + hunk_tokens
                else:
                    current_chunk += hunk_text
                    current_tokens += hunk_tokens
        else:
            if current_tokens + file_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = file_diff
                current_tokens = file_tokens
            else:
                current_chunk += "\n\n" + file_diff if current_chunk else file_diff
                current_tokens += file_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def review_chunk(groq_client: Groq, chunk: str, chunk_index: int, total_chunks: int) -> str:
    """Send a diff chunk to Groq for review."""
    prefix = f"""This is chunk {chunk_index + 1} of {total_chunks} from a pull request diff.
Review only the changes shown in this chunk. Focus on the actual diff (lines starting with + or -).

DIFF CHUNK:
```diff
{chunk}
```

Provide your structured review below:"""

    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prefix}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for chunk {chunk_index + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return f"**Error reviewing chunk {chunk_index + 1}**: Failed after {MAX_RETRIES} attempts."

    return ""


def aggregate_reviews(reviews: List[str]) -> str:
    """Combine multiple chunk reviews into a single structured comment."""
    if len(reviews) == 1:
        return reviews[0]

    # Combine all reviews and ask Groq to synthesize
    combined = "\n\n---\n\n".join(
        f"## Review Part {i + 1}\n\n{r}" 
        for i, r in enumerate(reviews) if r.strip()
    )

    synthesis_prompt = f"""You are a senior engineering lead. Below are code reviews from multiple chunks of the same PR.
Synthesize them into ONE cohesive review with the following structure:

## Summary
Overall assessment combining all parts.

## Issues Found

### High Severity
(List all high severity issues from all parts, deduplicate similar ones)

### Medium Severity
(List all medium severity issues, deduplicate)

### Low Severity
(List all low severity issues, deduplicate)

## Suggestions
(Combine and deduplicate general suggestions)

If a severity level has no issues, write "No issues found."

REVIEWS TO SYNTHESIZE:
{combined}"""

    return synthesis_prompt


def synthesize_reviews(groq_client: Groq, reviews: List[str]) -> str:
    """Use Groq to synthesize multiple chunk reviews into one."""
    if len(reviews) <= 1:
        return reviews[0] if reviews else "No review generated."

    synthesis_prompt = aggregate_reviews(reviews)

    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior engineering lead. Synthesize multiple code reviews into one cohesive, structured review. Be concise and deduplicate issues."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Synthesis attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                # Fallback: concatenate with headers
                return "\n\n".join(
                    f"### Part {i + 1}\n\n{r}" 
                    for i, r in enumerate(reviews) if r.strip()
                )

    return ""


def post_review_comment(github_client: Github, repo_name: str, pr_number: int, review: str) -> None:
    """Post the review as a PR comment."""
    repo = github_client.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    header = "## 🤖 AI Code Review\n\n"
    footer = "\n\n---\n*Generated by Groq (llama-3.3-70b-versatile) via GitHub Actions*"

    full_comment = header + review + footer

    # GitHub has a 65536 character limit for PR comments
    if len(full_comment) > 65000:
        full_comment = full_comment[:65000] + "\n\n...(truncated)"

    pr.create_issue_comment(full_comment)
    print(f"Posted review comment to PR #{pr_number}")


def main():
    """Main entry point."""
    print("Starting AI Code Review...")

    # Get configuration
    repo_name = get_env_var("REPO_NAME")
    pr_number = int(get_env_var("PR_NUMBER"))

    # Initialize clients
    github_client, groq_client = init_clients()

    # Fetch diff
    print(f"Fetching diff for PR #{pr_number} in {repo_name}...")
    diff_text = get_pr_diff(github_client, repo_name, pr_number)

    if not diff_text.strip():
        print("No diff found. Exiting.")
        post_review_comment(
            github_client, repo_name, pr_number,
            "No code changes detected in this PR."
        )
        return

    print(f"Diff size: {len(diff_text)} characters")

    # Chunk diff
    chunks = chunk_diff(diff_text)
    print(f"Split diff into {len(chunks)} chunk(s)")

    # Review each chunk
    reviews = []
    for i, chunk in enumerate(chunks):
        print(f"Reviewing chunk {i + 1}/{len(chunks)}...")
        review = review_chunk(groq_client, chunk, i, len(chunks))
        reviews.append(review)
        if i < len(chunks) - 1:
            time.sleep(1)  # Rate limit protection

    # Synthesize if multiple chunks
    if len(reviews) > 1:
        print("Synthesizing reviews...")
        final_review = synthesize_reviews(groq_client, reviews)
    else:
        final_review = reviews[0]

    # Post comment
    print("Posting review comment...")
    post_review_comment(github_client, repo_name, pr_number, final_review)

    print("AI Code Review complete!")


if __name__ == "__main__":
    main()
