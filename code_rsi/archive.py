"""Immutable, content-addressed source archives and attempt lineage (stdlib only).

Only explicitly supplied UTF-8 files and four metadata sections are stored/exported.
This module never walks/copies a host repository or follows filesystem links.

API::
    archive = ProgramArchive('/trusted/archive')
    node = archive.record({'main.py': 'print(1)\n'}, {}, session_id='trial', attempt=0)
    child = archive.apply_patch(node['node_id'], writes={'helper.py': 'VALUE=2\n'},
                                session_id='trial', attempt=1)
    archive.export_node(child['node_id'], '/new/nonexistent/export')

program_id hashes source contents and metadata, excluding lineage. node_id also
binds parent_node_id, session_id, and attempt. Duplicate programs are reused;
every distinct attempt has a node. Reusing the same node identity raises Conflict.
Exports contain only files/ and metadata.json; neither archive nodes nor logs.
Callers must not supply credentials in source text or metadata values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import stat
import unicodedata
from dataclasses import dataclass
from typing import Mapping


class ArchiveError(ValueError):
    """Invalid input or archive integrity failure."""


class Conflict(ArchiveError):
    """An immutable node or export destination already exists."""


@dataclass(frozen=True)
class Limits:
    max_files: int = 64
    max_file_bytes: int = 262144
    max_total_bytes: int = 2097152
    max_metadata_bytes: int = 262144

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ArchiveError('limits must be positive integers')


_METADATA = ('config', 'prompts', 'dependency_lock', 'index_builder')
_FORBIDDEN = {'.git', '.hg', '.svn', 'runs', 'logs', '.codex', '.ssh', '.aws',
              'credentials', 'credentials.json', 'secrets', 'secrets.json'}
_DEVICES = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} | {
    p + d for p in ('COM', 'LPT') for d in '123456789\u00b9\u00b2\u00b3'}
_ID = re.compile(r'^[0-9a-f]{64}$')


def _json_bytes(value):
    def check(v):
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float and math.isfinite(v):
            return
        if type(v) is list:
            for x in v:
                check(x)
            return
        if type(v) is dict and all(type(k) is str for k in v):
            for x in v.values():
                check(x)
            return
        raise ArchiveError('metadata must contain finite JSON values and string keys')
    check(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ArchiveError('invalid UTF-8 JSON') from exc


def _hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _identifier(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ArchiveError('invalid archive identifier')
    return value


def _relative(value):
    if type(value) is not str or not value or '\\' in value:
        raise ArchiveError('file names must be forward-slash relative paths')
    if pathlib.PurePosixPath(value).is_absolute() or pathlib.PureWindowsPath(value).drive:
        raise ArchiveError('absolute paths and drive names are forbidden')
    if len(value.encode('utf-8')) > 1024:
        raise ArchiveError('relative path exceeds 1024 UTF-8 bytes')
    parts = value.split('/')
    for part in parts:
        if not part or part in ('.', '..') or part.endswith((' ', '.')):
            raise ArchiveError('empty, dot, traversal, or trailing-dot/space path')
        if any(ord(c) < 32 or c in '<>:"|?*' for c in part):
            raise ArchiveError('unsupported Windows path character')
        if part.split('.')[0].upper() in _DEVICES:
            raise ArchiveError('Windows reserved device name')
        folded = unicodedata.normalize('NFC', part).casefold()
        if folded in _FORBIDDEN or folded == '.env' or folded.startswith('.env.'):
            raise ArchiveError('host state, logs, and credential paths are forbidden')
        if folded.endswith(('.pem', '.key')):
            raise ArchiveError('credential container extension is forbidden')
    return value


def _validate_paths(paths):
    aliases = {}
    full = set(paths)
    for name in paths:
        _relative(name)
        parts = name.split('/')
        for i in range(1, len(parts) + 1):
            prefix = '/'.join(parts[:i])
            folded = unicodedata.normalize('NFC', prefix).casefold()
            if folded in aliases and aliases[folded] != prefix:
                raise ArchiveError('case/Unicode path aliases are forbidden')
            aliases[folded] = prefix
            if i < len(parts) and prefix in full:
                raise ArchiveError('file/directory path collision')


def _safe(path):
    """Reject symlinks and Windows reparse points in every existing ancestor."""
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    for current in reversed((path,) + tuple(path.parents)):
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ArchiveError('symlink/reparse-point paths are forbidden')
    return path


def _read_json(path):
    path = _safe(path)
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4194304:
            raise ArchiveError('archive JSON must be a regular file under 4 MiB')
        def pairs(items):
            result = {}
            for k, v in items:
                if k in result:
                    raise ArchiveError('duplicate JSON key in archive')
                result[k] = v
            return result
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError('cannot read immutable archive record') from exc


def _write_new(path, data):
    path = _safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe(path.parent)
    try:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise Conflict('immutable file already exists') from exc


class ProgramArchive:
    """An explicit-source archive. Mutation methods return a verified node dict.

    metadata/metadata_updates allow config, prompts, dependency_lock, index_builder.
    Missing sections default to {}; updates replace whole named sections, never
    perform an ambiguous recursive merge. Patches are transactional with respect
    to input validation; interrupted filesystem writes fail closed on later reads.
    No method executes source code or modifies an existing program/node/export.
    """
    def __init__(self, root, *, limits=None):
        self.root = _safe(root)
        self.limits = limits or Limits()
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ('programs', 'nodes'):
            _safe(self.root / sub).mkdir(exist_ok=True)

    def _metadata(self, metadata):
        if type(metadata) is not dict or not set(metadata) <= set(_METADATA):
            raise ArchiveError('unknown metadata section')
        result = {k: metadata.get(k, {}) for k in _METADATA}
        raw = _json_bytes(result)
        if len(raw) > self.limits.max_metadata_bytes:
            raise ArchiveError('metadata byte limit exceeded')
        return json.loads(raw)

    def _files(self, files):
        if not isinstance(files, Mapping) or not files or len(files) > self.limits.max_files:
            raise ArchiveError('file count outside limits')
        _validate_paths(files)
        result = {}
        total = 0
        for path, source in files.items():
            if type(source) is not str:
                raise ArchiveError('every source must be a UTF-8 string')
            try:
                data = source.encode('utf-8')
            except UnicodeError as exc:
                raise ArchiveError('invalid UTF-8 source') from exc
            if len(data) > self.limits.max_file_bytes:
                raise ArchiveError('per-file byte limit exceeded')
            total += len(data)
            if total > self.limits.max_total_bytes:
                raise ArchiveError('total source byte limit exceeded')
            result[path] = source
        return dict(sorted(result.items()))

    def _manifest(self, files, metadata):
        return {'schema': 'code-rsi-program-v1', 'metadata': metadata,
                'files': {p: {'sha256': hashlib.sha256(s.encode('utf-8')).hexdigest(),
                              'bytes': len(s.encode('utf-8'))} for p, s in files.items()}}

    def load_program(self, program_id):
        """Return {program_id, files, metadata} after full on-disk verification."""
        program_id = _identifier(program_id)
        directory = _safe(self.root / 'programs' / program_id)
        manifest = _read_json(directory / 'manifest.json')
        if type(manifest) is not dict or set(manifest) != {'schema', 'metadata', 'files'}:
            raise ArchiveError('unexpected program manifest schema')
        if manifest['schema'] != 'code-rsi-program-v1' or _hash(manifest) != program_id:
            raise ArchiveError('program manifest hash mismatch')
        if type(manifest['files']) is not dict:
            raise ArchiveError('invalid file manifest')
        _validate_paths(manifest['files'])
        if not manifest['files'] or len(manifest['files']) > self.limits.max_files:
            raise ArchiveError('file count outside limits')
        expected = {'manifest.json'} | {'files/' + p for p in manifest['files']}
        actual = set()
        for current, dirs, names in os.walk(directory, followlinks=False):
            _safe(current)
            for entry in dirs + names:
                _safe(pathlib.Path(current) / entry)
            for name in names:
                actual.add((pathlib.Path(current) / name).relative_to(directory).as_posix())
        if actual != expected:
            raise ArchiveError('extra or missing files in immutable program')
        files = {}
        for name, entry in manifest['files'].items():
            path = _safe(directory / 'files' / name)
            if type(entry) is not dict or set(entry) != {'sha256', 'bytes'}:
                raise ArchiveError('invalid source hash record')
            if type(entry['bytes']) is not int or not 0 <= entry['bytes'] <= self.limits.max_file_bytes:
                raise ArchiveError('invalid source size record')
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != entry['bytes']:
                raise ArchiveError('source must be a regular file with expected size')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ArchiveError('source hash mismatch')
            try:
                files[name] = raw.decode('utf-8')
            except UnicodeError as exc:
                raise ArchiveError('source UTF-8 mismatch') from exc
        files = self._files(files)
        metadata = self._metadata(manifest['metadata'])
        if self._manifest(files, metadata) != manifest:
            raise ArchiveError('noncanonical manifest')
        return {'program_id': program_id, 'files': files, 'metadata': metadata}

    def load_node(self, node_id):
        """Validate node identity, referenced program and immediate parent record."""
        node_id = _identifier(node_id)
        node = _read_json(self.root / 'nodes' / (node_id + '.json'))
        expected = {'schema', 'program_id', 'parent_node_id', 'session_id', 'attempt'}
        if type(node) is not dict or set(node) != expected or node.get('schema') != 'code-rsi-node-v1':
            raise ArchiveError('unexpected node schema')
        if _hash(node) != node_id:
            raise ArchiveError('node hash mismatch')
        self._attempt(node['session_id'], node['attempt'])
        self.load_program(node['program_id'])
        if node['parent_node_id'] is not None:
            parent_id = _identifier(node['parent_node_id'])
            parent = _read_json(self.root / 'nodes' / (parent_id + '.json'))
            if _hash(parent) != parent_id:
                raise ArchiveError('parent node hash mismatch')
        return {'node_id': node_id, **node}

    @staticmethod
    def _attempt(session_id, attempt):
        if type(session_id) is not str or not session_id.strip():
            raise ArchiveError('session_id must be a nonempty UTF-8 string up to 256 bytes')
        try:
            size = len(session_id.encode('utf-8'))
        except UnicodeError as exc:
            raise ArchiveError('invalid UTF-8 session_id') from exc
        if size > 256:
            raise ArchiveError('session_id must be a nonempty UTF-8 string up to 256 bytes')
        if type(attempt) is not int or attempt < 0:
            raise ArchiveError('attempt must be a nonnegative integer')

    def record(self, files, metadata, *, session_id, attempt, parent_node_id=None):
        """Create immutable program content (deduplicated) and a new attempt node."""
        files = self._files(files)
        metadata = self._metadata(metadata)
        self._attempt(session_id, attempt)
        if parent_node_id is not None:
            self.load_node(parent_node_id)
        manifest = self._manifest(files, metadata)
        program_id = _hash(manifest)
        node = {'schema': 'code-rsi-node-v1', 'program_id': program_id,
                'parent_node_id': parent_node_id, 'session_id': session_id, 'attempt': attempt}
        node_id = _hash(node)
        node_path = _safe(self.root / 'nodes' / (node_id + '.json'))
        if node_path.exists():
            raise Conflict('attempt node already exists')
        directory = _safe(self.root / 'programs' / program_id)
        if directory.exists():
            self.load_program(program_id)
        else:
            directory.mkdir()
            for name, source in files.items():
                _write_new(directory / 'files' / name, source.encode('utf-8'))
            _write_new(directory / 'manifest.json', _json_bytes(manifest))
        _write_new(node_path, _json_bytes(node))
        return self.load_node(node_id)

    def apply_patch(self, parent_node_id, *, writes=None, deletes=(), metadata_updates=None,
                    session_id, attempt):
        """Inherit a verified parent, apply explicit writes/deletes, create an attempt.

        Deleting absent files or writing and deleting the same path is an error.
        A no-op patch creates a new node but shares its parent's program_id.
        """
        parent = self.load_node(parent_node_id)
        program = self.load_program(parent['program_id'])
        writes = {} if writes is None else writes
        if not isinstance(writes, Mapping) or not isinstance(deletes, (tuple, list)):
            raise ArchiveError('writes must be a mapping and deletes a sequence')
        _validate_paths(writes)
        _validate_paths(deletes)
        if len(set(deletes)) != len(deletes) or set(writes) & set(deletes):
            raise ArchiveError('duplicate or contradictory patch operations')
        files = dict(program['files'])
        for path in deletes:
            if path not in files:
                raise ArchiveError('deletion target does not exist')
            del files[path]
        files.update(writes)
        updates = {} if metadata_updates is None else metadata_updates
        if type(updates) is not dict or not set(updates) <= set(_METADATA):
            raise ArchiveError('unknown metadata update section')
        metadata = {**program['metadata'], **updates}
        return self.record(files, metadata, session_id=session_id, attempt=attempt,
                           parent_node_id=parent_node_id)

    def export_node(self, node_id, destination):
        """Export only verified sources under files/ and metadata.json to a NEW dir."""
        node = self.load_node(node_id)
        program = self.load_program(node['program_id'])
        destination = _safe(destination)
        if destination.exists():
            raise Conflict('export destination must not exist')
        # Never place exports inside this archive, or place an archive inside export.
        if destination == self.root or self.root in destination.parents or destination in self.root.parents:
            raise ArchiveError('export and archive paths must be disjoint')
        destination.mkdir(parents=True)
        for path, source in program['files'].items():
            _write_new(destination / 'files' / path, source.encode('utf-8'))
        _write_new(destination / 'metadata.json', _json_bytes(program['metadata']))
        return destination
