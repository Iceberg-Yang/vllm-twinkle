#!/usr/bin/env python3
"""Offline publication checks, using only the standard library.

Does not execute notebooks, install packages, contact services, or prove GPU
compatibility. Link checks cover local file targets, not remote URLs or anchors.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', '.venv', '.pytest_cache', '.ipynb_checkpoints'}
SECRET_RULES = {
    'ModelScope token': re.compile(r'\bms-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b', re.I),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'Hugging Face token': re.compile(r'\bhf_[A-Za-z0-9]{25,}\b'),
    'Private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
TEXT_EXTS = {'.py', '.md', '.ipynb', '.json', '.yaml', '.yml', '.sh', '.txt', '.toml'}
errors: list[str] = []
counts = {'files': 0, 'python': 0, 'notebooks': 0, 'notebook_python_cells': 0,
          'magic_or_shell_cells': 0, 'local_links': 0, 'shell': 0}


def fail(path: Path, reason: str) -> None:
    errors.append(f'{path.relative_to(ROOT)}: {reason}')


def check_links(path: Path, text: str) -> None:
    # Fenced examples are not rendered links. Inline code examples may contain
    # protocol notation and are removed before checking Markdown link targets.
    text = re.sub(r'^(```|~~~).*?^\1\s*$', '', text, flags=re.M | re.S)
    text = re.sub(r'`[^`\n]*`', '', text)
    targets = re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', text)
    targets += re.findall(r'(?:src|href)=[\"\']([^\"\']+)[\"\']', text)
    for value in targets:
        target = value.strip().split(' "', 1)[0].strip('<>')
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if parsed.path.startswith('/'):
            fail(path, 'absolute local link; use a repository-relative target')
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(ROOT):
            fail(path, 'local link escapes the repository')
        elif not resolved.exists():
            fail(path, f'missing local link target: {target}')
        else:
            counts['local_links'] += 1


def check_notebook(path: Path, text: str) -> None:
    try:
        notebook = json.loads(text)
        if notebook.get('nbformat') != 4 or not isinstance(notebook.get('cells'), list):
            raise ValueError('expected nbformat 4 and a cells list')
        counts['notebooks'] += 1
        for index, cell in enumerate(notebook['cells']):
            source = cell.get('source', '')
            source = ''.join(source) if isinstance(source, list) else source
            if cell.get('cell_type') == 'markdown':
                check_links(path, source)
                continue
            if cell.get('cell_type') != 'code':
                continue
            if cell.get('outputs') or cell.get('execution_count') is not None:
                fail(path, f'cell {index}: stale outputs or execution count')
            if source.lstrip().startswith('%%'):
                counts['magic_or_shell_cells'] += 1
                first, _, body = source.partition('\n')
                if first.strip() in ('%%bash', '%%sh'):
                    result = subprocess.run(['bash', '-n'], input=body, text=True, capture_output=True)
                    if result.returncode:
                        fail(path, f'cell {index}: shell syntax error')
                # Other cell magics have their own language; no execution here.
                continue
            lines = source.splitlines()
            has_magic = False
            continuation = False
            cleaned = []
            for line in lines:
                stripped = line.lstrip()
                if continuation or stripped.startswith(('!', '%', '?')):
                    has_magic = True
                    continuation = line.rstrip().endswith('\\')
                    cleaned.append(' ' * (len(line) - len(stripped)) + 'pass')
                else:
                    cleaned.append(line)
            if has_magic:
                counts['magic_or_shell_cells'] += 1
            try:
                ast.parse('\n'.join(cleaned), filename=f'{path.name}:cell-{index}')
                counts['notebook_python_cells'] += 1
            except SyntaxError as exc:
                fail(path, f'cell {index}: Python syntax error at line {exc.lineno}: {exc.msg}')
    except (ValueError, TypeError, KeyError) as exc:
        fail(path, f'invalid Notebook structure: {exc}')


def main() -> int:
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.startswith('.venv') for part in relative.parts):
            continue
        if path.is_symlink():
            fail(path, 'symlink requires explicit review')
            continue
        if not path.is_file():
            continue
        counts['files'] += 1
        if path.name == '.DS_Store':
            fail(path, 'operating-system metadata must not be included')
        if path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example'):
            fail(path, 'local environment file must not be included')
            continue  # Do not print or inspect real credentials.
        if path.stat().st_size > 25 * 1024**2:
            fail(path, 'file exceeds 25 MiB; review distribution method')
        if path.suffix not in TEXT_EXTS and path.name not in {'Dockerfile', '.env.example', '.gitignore'}:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            fail(path, 'expected UTF-8 text')
            continue
        for category, pattern in SECRET_RULES.items():
            if pattern.search(text):
                fail(path, f'possible {category}; value intentionally not printed')
        private_home = '/' + 'Users/'
        if private_home in text or 'file' + ':///' in text:
            fail(path, 'personal absolute path or file URL')
        if re.search(r'twinkle://\d{8}_', text):
            fail(path, 'personal historical checkpoint identifier')
        if path.name == '.env.example':
            for line in text.splitlines():
                if line.startswith(('MODELSCOPE_TOKEN=', 'TWINKLE_MODEL_PATH=')) and line.partition('=')[2].strip():
                    fail(path, 'example credential/checkpoint must be empty')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=str(relative))
                counts['python'] += 1
            except SyntaxError as exc:
                fail(path, f'Python syntax error at line {exc.lineno}: {exc.msg}')
        elif path.suffix == '.ipynb':
            check_notebook(path, text)
        elif path.suffix == '.md':
            check_links(path, text)
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            counts['shell'] += 1
            if result.returncode:
                fail(path, 'shell syntax error')
        elif path.suffix == '.json':
            try:
                json.loads(text)
            except ValueError:
                fail(path, 'invalid JSON')
    for message in errors:
        print('FAIL', message)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    print('PASS: offline checks only' if not errors else f'FAILED: {len(errors)} issue(s)')
    print('Not checked: external URLs, Markdown anchors, GPU execution, Docker build, TaaS access, image rights.')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
