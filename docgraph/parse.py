"""Universal tree-sitter parser.

Each language ships its own pip package; we import lazily so missing
languages don't break the rest. Adding a new language = pip install
tree-sitter-<lang> and add an entry to LANGUAGES below.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import tree_sitter as ts

log = logging.getLogger(__name__)


# (extension → language key)
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


# (language key → (module_name, language_function_name_or_attr))
# language_function_name supports either a callable like `language()` or
# `language_typescript()` / `language_tsx()` for the multi-language packages.
LANGUAGES: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "bash": ("tree_sitter_bash", "language"),
    "html": ("tree_sitter_html", "language"),
    "css": ("tree_sitter_css", "language"),
    "json": ("tree_sitter_json", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
}


# Tags queries — the standard @definition.X / @reference.X capture convention.
TAGS_QUERIES: dict[str, str] = {
    "python": """
(function_definition name: (identifier) @name) @definition.function
(class_definition name: (identifier) @name) @definition.class
(class_definition
  superclasses: (argument_list (identifier) @parent.class))
(decorator (identifier) @decorator.name)
(call function: (identifier) @ref.call)
(call function: (attribute attribute: (identifier) @ref.call))
(import_statement name: (dotted_name) @import.module)
(import_from_statement module_name: (dotted_name) @import.module)
""",
    "javascript": """
(function_declaration name: (identifier) @name) @definition.function
(method_definition name: (property_identifier) @name) @definition.method
(class_declaration name: (identifier) @name) @definition.class
(class_declaration (class_heritage (identifier) @parent.class))
(variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function
(call_expression function: (identifier) @ref.call)
(call_expression function: (member_expression property: (property_identifier) @ref.call))
(new_expression constructor: (identifier) @ref.new)
(import_statement source: (string) @import.module)
""",
    "typescript": """
(function_declaration name: (identifier) @name) @definition.function
(method_definition name: (property_identifier) @name) @definition.method
(class_declaration name: (type_identifier) @name) @definition.class
(class_declaration (class_heritage (extends_clause value: (identifier) @parent.class)))
(interface_declaration name: (type_identifier) @name) @definition.interface
(variable_declarator
  name: (identifier) @name
  value: [(arrow_function) (function_expression)]) @definition.function
(call_expression function: (identifier) @ref.call)
(call_expression function: (member_expression property: (property_identifier) @ref.call))
(new_expression constructor: (identifier) @ref.new)
(import_statement source: (string) @import.module)
""",
    "java": """
(method_declaration name: (identifier) @name) @definition.method
(class_declaration name: (identifier) @name) @definition.class
(class_declaration (superclass (type_identifier) @parent.class))
(interface_declaration name: (identifier) @name) @definition.interface
(method_invocation name: (identifier) @ref.call)
(object_creation_expression type: (type_identifier) @ref.new)
(import_declaration (scoped_identifier) @import.module)
""",
    "go": """
(function_declaration name: (identifier) @name) @definition.function
(method_declaration name: (field_identifier) @name) @definition.method
(type_declaration (type_spec name: (type_identifier) @name)) @definition.class
(call_expression function: (identifier) @ref.call)
(call_expression function: (selector_expression field: (field_identifier) @ref.call))
(import_spec path: (interpreted_string_literal) @import.module)
""",
    "rust": """
(function_item name: (identifier) @name) @definition.function
(struct_item name: (type_identifier) @name) @definition.class
(enum_item name: (type_identifier) @name) @definition.class
(trait_item name: (type_identifier) @name) @definition.interface
(call_expression function: (identifier) @ref.call)
(call_expression function: (field_expression field: (field_identifier) @ref.call))
""",
    "c": """
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function
(call_expression function: (identifier) @ref.call)
""",
    "cpp": """
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function
(class_specifier name: (type_identifier) @name) @definition.class
(struct_specifier name: (type_identifier) @name) @definition.class
(call_expression function: (identifier) @ref.call)
""",
    "c_sharp": """
(method_declaration name: (identifier) @name) @definition.method
(class_declaration name: (identifier) @name) @definition.class
(interface_declaration name: (identifier) @name) @definition.interface
(invocation_expression function: (identifier) @ref.call)
(invocation_expression function: (member_access_expression name: (identifier) @ref.call))
(object_creation_expression type: (identifier) @ref.new)
(using_directive (qualified_name) @import.module)
""",
    "ruby": """
(method name: (identifier) @name) @definition.method
(class name: (constant) @name) @definition.class
(module name: (constant) @name) @definition.class
(call method: (identifier) @ref.call)
""",
    "php": """
(function_definition name: (name) @name) @definition.function
(method_declaration name: (name) @name) @definition.method
(class_declaration name: (name) @name) @definition.class
(function_call_expression function: (name) @ref.call)
""",
    "bash": """
(function_definition name: (word) @name) @definition.function
(command name: (command_name (word) @ref.call))
""",
    "html": """
(element (start_tag (tag_name) @ref.call))
""",
    "css": """
(rule_set (selectors (class_selector (class_name) @name))) @definition.class
""",
    "json": "",
    "yaml": "",
}


@dataclass
class Entity:
    kind: str
    name: str
    qname: str
    file: str
    line_start: int
    line_end: int
    body: str = ""
    signature: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class RawEdge:
    kind: str
    src_file: str
    src_qname: str | None
    target_name: str
    line: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class FileParse:
    file: str
    language: str
    lines: int
    entities: list[Entity]
    edges: list[RawEdge]


@lru_cache(maxsize=64)
def _load_language(lang_key: str) -> ts.Language | None:
    if lang_key not in LANGUAGES:
        return None
    mod_name, fn_name = LANGUAGES[lang_key]
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        return ts.Language(fn())
    except Exception as e:
        log.debug(f"failed to load {lang_key}: {e}")
        return None


@lru_cache(maxsize=64)
def _get_parser(lang_key: str) -> ts.Parser | None:
    lang = _load_language(lang_key)
    if lang is None:
        return None
    return ts.Parser(lang)


@lru_cache(maxsize=64)
def _get_query(lang_key: str) -> ts.Query | None:
    lang = _load_language(lang_key)
    if lang is None:
        return None
    src = TAGS_QUERIES.get(lang_key, "")
    if not src.strip():
        return None
    try:
        return ts.Query(lang, src)
    except Exception as e:
        log.debug(f"query compile failed for {lang_key}: {e}")
        return None


def detect_language(path: Path) -> str | None:
    return EXT_TO_LANG.get(path.suffix.lower())


def _capture_dict(query: ts.Query, root: ts.Node) -> dict[str, list[ts.Node]]:
    cur = ts.QueryCursor(query)
    return cur.captures(root)


def _enclosing(node: ts.Node, defs: list[tuple[ts.Node, str, str]]) -> tuple[str, str] | None:
    best = None
    best_size = float("inf")
    s, e = node.start_byte, node.end_byte
    for d_node, qname, kind in defs:
        if d_node.start_byte <= s and d_node.end_byte >= e:
            size = d_node.end_byte - d_node.start_byte
            if size < best_size:
                best_size = size
                best = (qname, kind)
    return best


DEF_KIND_MAP = {
    "definition.function": "function",
    "definition.method": "method",
    "definition.class": "class",
    "definition.interface": "interface",
}


def parse_file(path: Path, repo_root: Path) -> FileParse | None:
    lang_key = detect_language(path)
    if lang_key is None:
        return None
    parser = _get_parser(lang_key)
    if parser is None:
        return None
    try:
        source = path.read_bytes()
    except OSError:
        return None
    try:
        tree = parser.parse(source)
    except Exception:
        return None
    rel = str(path.relative_to(repo_root)).replace("\\", "/")
    lines = source.count(b"\n") + 1

    entities: list[Entity] = []
    raw_edges: list[RawEdge] = []
    defs: list[tuple[ts.Node, str, str]] = []  # (node, qname, kind)

    query = _get_query(lang_key)
    if query is None:
        # Just a File node, no entities
        return FileParse(file=rel, language=lang_key, lines=lines, entities=[], edges=[])

    caps = _capture_dict(query, tree.root_node)

    name_nodes = caps.get("name", [])

    # Build definitions by pairing each @definition.X with its enclosed @name
    for cap_name, kind in DEF_KIND_MAP.items():
        for d_node in caps.get(cap_name, []):
            # Find smallest @name child
            name_node = None
            for n in name_nodes:
                if d_node.start_byte <= n.start_byte and d_node.end_byte >= n.end_byte:
                    if name_node is None or n.start_byte < name_node.start_byte:
                        name_node = n
            if name_node is None:
                continue
            name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
            qname = f"{rel}::{name}"
            entities.append(Entity(
                kind=kind,
                name=name,
                qname=qname,
                file=rel,
                line_start=d_node.start_point.row + 1,
                line_end=d_node.end_point.row + 1,
                body=source[d_node.start_byte:d_node.end_byte][:8000].decode("utf-8", errors="replace"),
            ))
            defs.append((d_node, qname, kind))

    # Re-scope methods inside classes
    qname_remap: dict[str, str] = {}
    for d_node, qname, kind in defs:
        if kind in ("class", "interface"):
            continue
        # Find smallest enclosing class
        enclosing_class = None
        for d2, q2, k2 in defs:
            if d2 is d_node:
                continue
            if k2 not in ("class", "interface"):
                continue
            if d2.start_byte <= d_node.start_byte and d2.end_byte >= d_node.end_byte:
                if enclosing_class is None or (d2.end_byte - d2.start_byte) < enclosing_class[1]:
                    enclosing_class = (q2, d2.end_byte - d2.start_byte)
        if enclosing_class:
            new_q = f"{enclosing_class[0]}::{qname.split('::')[-1]}"
            qname_remap[qname] = new_q

    if qname_remap:
        for ent in entities:
            if ent.qname in qname_remap:
                ent.qname = qname_remap[ent.qname]
        defs = [(n, qname_remap.get(q, q), k) for (n, q, k) in defs]

    # Edges: refs
    for ref_cap, edge_kind in [("ref.call", "CALLS"), ("ref.new", "INSTANTIATES")]:
        for r_node in caps.get(ref_cap, []):
            target = source[r_node.start_byte:r_node.end_byte].decode("utf-8", errors="replace")
            enc = _enclosing(r_node, defs)
            raw_edges.append(RawEdge(
                kind=edge_kind,
                src_file=rel,
                src_qname=enc[0] if enc else None,
                target_name=target,
                line=r_node.start_point.row + 1,
            ))

    # Inheritance
    for p_node in caps.get("parent.class", []):
        enc = _enclosing(p_node, defs)
        if enc and enc[1] in ("class", "interface"):
            raw_edges.append(RawEdge(
                kind="INHERITS",
                src_file=rel,
                src_qname=enc[0],
                target_name=source[p_node.start_byte:p_node.end_byte].decode("utf-8", errors="replace"),
            ))

    # Decorators
    for d_node in caps.get("decorator.name", []):
        enc = _enclosing(d_node, defs)
        if enc:
            raw_edges.append(RawEdge(
                kind="DECORATED_BY",
                src_file=rel,
                src_qname=enc[0],
                target_name=source[d_node.start_byte:d_node.end_byte].decode("utf-8", errors="replace"),
            ))

    # Imports
    for i_node in caps.get("import.module", []):
        mod = source[i_node.start_byte:i_node.end_byte].decode("utf-8", errors="replace").strip("'\"<>")
        raw_edges.append(RawEdge(
            kind="IMPORTS",
            src_file=rel,
            src_qname=None,
            target_name=mod,
        ))

    return FileParse(
        file=rel,
        language=lang_key,
        lines=lines,
        entities=entities,
        edges=raw_edges,
    )
