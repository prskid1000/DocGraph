"""Universal tree-sitter parser.

Each language ships its own pip package; we import lazily so missing
languages don't break the rest. Adding a new language = pip install
tree-sitter-<lang> and add an entry to LANGUAGES below.
"""
from __future__ import annotations

import importlib
import logging
import re
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
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".ex": "elixir",
    ".exs": "elixir",
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
    ".md": "markdown",
    ".markdown": "markdown",
    ".swift": "swift",
    ".dart": "dart",
    ".lua": "lua",
    ".zig": "zig",
    ".zon": "zig",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml_interface",
    ".jl": "julia",
    ".pl": "perl",
    ".pm": "perl",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".m": "objc",
    ".mm": "objc",
    ".sql": "sql",
    ".toml": "toml",
    ".xml": "xml",
    ".xsd": "xml",
    ".xsl": "xml",
    ".xslt": "xml",
    ".plist": "xml",
    ".csproj": "xml",
    ".vbproj": "xml",
    ".fsproj": "xml",
    ".vcxproj": "xml",
    ".props": "xml",
    ".targets": "xml",
    ".pom": "xml",
    ".nuspec": "xml",
    ".resx": "xml",
    ".wsdl": "xml",
    ".xaml": "xml",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".hcl": "hcl",
    ".mk": "make",
    ".mak": "make",
    ".svelte": "svelte",
    ".nix": "nix",
    ".groovy": "groovy",
    ".gvy": "groovy",
    ".gradle": "groovy",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f08": "fortran",
    ".f": "fortran",
    ".for": "fortran",
    ".ftn": "fortran",
    ".v": "verilog",
    ".vh": "verilog",
    ".sv": "verilog",
    ".svh": "verilog",
    ".vhd": "vhdl",
    ".vhdl": "vhdl",
}


# Exact lower-case basenames that name a grammar language regardless of
# extension (Dockerfile, Makefile, ...). Checked before EXT_TO_LANG.
FILENAME_TO_LANG: dict[str, str] = {
    "makefile": "make",
    "gnumakefile": "make",
    "bsdmakefile": "make",
    "jenkinsfile": "groovy",
    "cargo.lock": "toml",
    "poetry.lock": "toml",
    "uv.lock": "toml",
}

# Lower-case basename prefixes (e.g. "dockerfile." for `Dockerfile.dev`).
FILENAME_PREFIX_TO_LANG: dict[str, str] = {
    "makefile.": "make",
    "jenkinsfile.": "groovy",
}


# --- Plain-text fallback ----------------------------------------------------
# Files no grammar claims are still indexed when they are text: a File node
# whose `language` is one of these kinds (or "text"), plus line/paragraph
# chunks that are embedded and keyword-searchable. A grammar entry above
# always wins over these tables.
TEXT_EXT_KINDS: dict[str, str] = {
    ".txt": "txt", ".text": "txt", ".rst": "rst", ".adoc": "asciidoc",
    ".asciidoc": "asciidoc", ".org": "org", ".tex": "latex", ".mdx": "markdown",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini", ".properties": "ini",
    ".editorconfig": "ini", ".toml": "toml", ".xml": "xml", ".xsd": "xml",
    ".xsl": "xml", ".plist": "xml", ".csproj": "xml", ".vcxproj": "xml",
    ".props": "xml", ".targets": "xml", ".pom": "xml", ".svg": "xml",
    ".sql": "sql", ".csv": "csv", ".tsv": "csv", ".graphql": "graphql",
    ".gql": "graphql", ".proto": "protobuf", ".tf": "hcl", ".tfvars": "hcl",
    ".hcl": "hcl", ".nix": "nix", ".cmake": "cmake", ".mk": "makefile",
    ".mak": "makefile", ".dockerfile": "dockerfile", ".gradle": "groovy",
    ".groovy": "groovy", ".bat": "batch", ".cmd": "batch", ".ps1": "powershell",
    ".psm1": "powershell", ".psd1": "powershell", ".vue": "vue",
    ".svelte": "svelte", ".lua": "lua", ".swift": "swift", ".dart": "dart",
    ".zig": "zig", ".hs": "haskell", ".ml": "ocaml", ".mli": "ocaml",
    ".r": "r", ".jl": "julia", ".pl": "perl", ".pm": "perl", ".m": "objc",
    ".mm": "objc", ".erl": "erlang", ".hrl": "erlang", ".clj": "clojure",
    ".cljs": "clojure", ".cljc": "clojure", ".edn": "clojure", ".f90": "fortran",
    ".f95": "fortran", ".f": "fortran", ".for": "fortran", ".v": "verilog",
    ".sv": "verilog", ".vhd": "vhdl", ".vhdl": "vhdl", ".env": "env",
    ".example": "env-example", ".sample": "env-example", ".template": "text",
    ".log": "text", ".diff": "diff", ".patch": "diff", ".jsonl": "json",
    ".json5": "json", ".jsonc": "json", ".ndjson": "json", ".lock": "text",
    ".gitignore": "ignore", ".dockerignore": "ignore", ".npmignore": "ignore",
    ".htaccess": "ini", ".service": "ini", ".desktop": "ini", ".rc": "text",
    ".man": "text", ".1": "text", ".nfo": "txt", ".srt": "txt", ".vtt": "txt",
}

# Lower-case basenames (no or odd extension) -> text kind.
TEXT_NAME_KINDS: dict[str, str] = {
    "dockerfile": "dockerfile", "containerfile": "dockerfile",
    "makefile": "makefile", "gnumakefile": "makefile", "bsdmakefile": "makefile",
    "jenkinsfile": "groovy", "vagrantfile": "ruby", "gemfile": "ruby",
    "rakefile": "ruby", "podfile": "ruby", "brewfile": "ruby", "procfile": "ini",
    "justfile": "makefile", "cmakelists.txt": "cmake", "build": "text",
    "workspace": "text", "license": "txt", "licence": "txt", "copying": "txt",
    "notice": "txt", "authors": "txt", "contributors": "txt", "readme": "txt",
    "changelog": "txt", "changes": "txt", "history": "txt", "news": "txt",
    "todo": "txt", "codeowners": "text", "owners": "text", "version": "text",
    ".env.example": "env-example", ".env.sample": "env-example",
    ".env.template": "env-example", "env.example": "env-example",
    ".editorconfig": "ini", ".gitattributes": "ignore",
}

# Lower-case basename prefixes for text kinds (`Dockerfile.prod`, `Makefile.win`).
TEXT_NAME_PREFIX_KINDS: dict[str, str] = {
    "dockerfile.": "dockerfile", "containerfile.": "dockerfile",
    "makefile.": "makefile", "jenkinsfile.": "groovy", "license.": "txt",
    "licence.": "txt", "readme.": "txt", "changelog.": "txt",
    "requirements": "text",
}

# Extensions that are never text, whatever the sniff says (the ignore list
# already drops most media; this guards files that slipped past it).
BINARY_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".ico", ".webp",
    ".psd", ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".avi", ".mov", ".mkv",
    ".webm", ".flv", ".ttf", ".otf", ".woff", ".woff2", ".eot", ".pdf", ".doc",
    ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".zip", ".gz", ".tgz",
    ".bz2", ".xz", ".7z", ".rar", ".tar", ".jar", ".war", ".exe", ".dll", ".so",
    ".dylib", ".bin", ".o", ".a", ".lib", ".obj", ".class", ".pyc", ".pyo",
    ".pyd", ".wasm", ".db", ".sqlite", ".sqlite3", ".kuzu", ".npy", ".npz",
    ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".safetensors", ".h5", ".hdf5",
    ".parquet", ".arrow", ".feather", ".avro", ".orc", ".dat", ".iso", ".img",
    ".dmg", ".msi", ".apk", ".aab", ".ipa", ".deb", ".rpm", ".beam", ".swf",
})

SNIFF_BYTES = 8192


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
    "kotlin": ("tree_sitter_kotlin", "language"),
    "scala": ("tree_sitter_scala", "language"),
    "elixir": ("tree_sitter_elixir", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "bash": ("tree_sitter_bash", "language"),
    "html": ("tree_sitter_html", "language"),
    "css": ("tree_sitter_css", "language"),
    "json": ("tree_sitter_json", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
    "markdown": ("tree_sitter_markdown", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "dart": ("tree_sitter_dart", "language"),
    "lua": ("tree_sitter_lua", "language"),
    "zig": ("tree_sitter_zig", "language"),
    "haskell": ("tree_sitter_haskell", "language"),
    "ocaml": ("tree_sitter_ocaml", "language_ocaml"),
    "ocaml_interface": ("tree_sitter_ocaml", "language_ocaml_interface"),
    "julia": ("tree_sitter_julia", "language"),
    "perl": ("tree_sitter_perl", "language"),
    "powershell": ("tree_sitter_powershell", "language"),
    "objc": ("tree_sitter_objc", "language"),
    "sql": ("tree_sitter_sql", "language"),
    "toml": ("tree_sitter_toml", "language"),
    "xml": ("tree_sitter_xml", "language_xml"),
    "hcl": ("tree_sitter_hcl", "language"),
    "make": ("tree_sitter_make", "language"),
    "svelte": ("tree_sitter_svelte", "language"),
    "nix": ("tree_sitter_nix", "language"),
    "groovy": ("tree_sitter_groovy", "language"),
    "fortran": ("tree_sitter_fortran", "language"),
    "verilog": ("tree_sitter_verilog", "language"),
    "vhdl": ("tree_sitter_vhdl", "language"),
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
(import_from_statement
  module_name: (dotted_name) @import.module
  name: (dotted_name) @import.symbol)
(import_from_statement
  module_name: (dotted_name) @import.module
  name: (aliased_import name: (dotted_name) @import.symbol))
(module
  (expression_statement
    (assignment left: (identifier) @name)) @definition.variable)
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
(import_specifier name: (identifier) @import.symbol)
(import_clause (identifier) @import.symbol)
(program
  (lexical_declaration
    (variable_declarator name: (identifier) @name) @definition.variable))
(program
  (variable_declaration
    (variable_declarator name: (identifier) @name) @definition.variable))
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
(import_specifier name: (identifier) @import.symbol)
(import_clause (identifier) @import.symbol)
(program
  (lexical_declaration
    (variable_declarator name: (identifier) @name) @definition.variable))
(program
  (variable_declaration
    (variable_declarator name: (identifier) @name) @definition.variable))
(public_field_definition name: (property_identifier) @name) @definition.variable
""",
    "java": """
(method_declaration name: (identifier) @name) @definition.method
(class_declaration name: (identifier) @name) @definition.class
(class_declaration (superclass (type_identifier) @parent.class))
(interface_declaration name: (identifier) @name) @definition.interface
(method_invocation name: (identifier) @ref.call)
(object_creation_expression type: (type_identifier) @ref.new)
(import_declaration (scoped_identifier) @import.module)
(field_declaration
  declarator: (variable_declarator name: (identifier) @name) @definition.variable)
""",
    "go": """
(function_declaration name: (identifier) @name) @definition.function
(method_declaration name: (field_identifier) @name) @definition.method
(type_declaration (type_spec name: (type_identifier) @name)) @definition.class
(call_expression function: (identifier) @ref.call)
(call_expression function: (selector_expression field: (field_identifier) @ref.call))
(import_spec path: (interpreted_string_literal) @import.module)
(source_file
  (var_declaration
    (var_spec name: (identifier) @name) @definition.variable))
(source_file
  (const_declaration
    (const_spec name: (identifier) @name) @definition.variable))
""",
    "rust": """
(function_item name: (identifier) @name) @definition.function
(struct_item name: (type_identifier) @name) @definition.class
(enum_item name: (type_identifier) @name) @definition.class
(trait_item name: (type_identifier) @name) @definition.interface
(call_expression function: (identifier) @ref.call)
(call_expression function: (field_expression field: (field_identifier) @ref.call))
(source_file (const_item name: (identifier) @name) @definition.variable)
(source_file (static_item name: (identifier) @name) @definition.variable)
""",
    "c": """
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function
(call_expression function: (identifier) @ref.call)
(translation_unit
  (declaration
    declarator: (init_declarator declarator: (identifier) @name) @definition.variable))
""",
    "cpp": """
(function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function
(class_specifier name: (type_identifier) @name) @definition.class
(struct_specifier name: (type_identifier) @name) @definition.class
(call_expression function: (identifier) @ref.call)
(translation_unit
  (declaration
    declarator: (init_declarator declarator: (identifier) @name) @definition.variable))
(field_declaration
  declarator: (field_identifier) @name) @definition.variable
""",
    "c_sharp": """
(method_declaration name: (identifier) @name) @definition.method
(class_declaration name: (identifier) @name) @definition.class
(interface_declaration name: (identifier) @name) @definition.interface
(invocation_expression function: (identifier) @ref.call)
(invocation_expression function: (member_access_expression name: (identifier) @ref.call))
(object_creation_expression type: (identifier) @ref.new)
(using_directive (qualified_name) @import.module)
(field_declaration
  (variable_declaration
    (variable_declarator name: (identifier) @name) @definition.variable))
(property_declaration name: (identifier) @name) @definition.variable
""",
    "kotlin": """
(function_declaration (identifier) @name) @definition.function
(class_declaration (identifier) @name) @definition.class
(object_declaration (identifier) @name) @definition.class
(property_declaration (variable_declaration (identifier) @name)) @definition.variable
(call_expression (identifier) @ref.call)
(import (qualified_identifier) @import.module)
""",
    "scala": """
(function_definition (identifier) @name) @definition.function
(class_definition (identifier) @name) @definition.class
(object_definition (identifier) @name) @definition.class
(trait_definition (identifier) @name) @definition.interface
(val_definition (identifier) @name) @definition.variable
(var_definition (identifier) @name) @definition.variable
(call_expression (identifier) @ref.call)
(import_declaration (identifier) @import.module)
""",
    "elixir": """
((call
  target: (identifier) @_kw
  (arguments (alias) @name)) @definition.class
 (#any-of? @_kw "defmodule" "defprotocol" "defimpl"))
((call
  target: (identifier) @_kw
  (arguments (call target: (identifier) @name))) @definition.function
 (#any-of? @_kw "def" "defp" "defmacro" "defmacrop"))
((call
  target: (identifier) @_kw
  (arguments (identifier) @name)) @definition.function
 (#any-of? @_kw "def" "defp" "defmacro" "defmacrop"))
""",
    "ruby": """
(method name: (identifier) @name) @definition.method
(class name: (constant) @name) @definition.class
(module name: (constant) @name) @definition.class
(call method: (identifier) @ref.call)
(program (assignment left: (constant) @name) @definition.variable)
(program (assignment left: (global_variable) @name) @definition.variable)
""",
    "php": """
(function_definition name: (name) @name) @definition.function
(method_declaration name: (name) @name) @definition.method
(class_declaration name: (name) @name) @definition.class
(function_call_expression function: (name) @ref.call)
(property_declaration
  (property_element (variable_name (name) @name)) @definition.variable)
(const_declaration
  (const_element (name) @name) @definition.variable)
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
((declaration (property_name) @name) @definition.variable
 (#match? @name "^--"))
""",
    "json": "",
    "yaml": "",
    # Headings become "section" entities. tree-sitter-markdown uses `inline`
    # for the heading text in ATX headings (# style) and `paragraph` wrapping
    # `inline` for setext headings (underlined style).
    "markdown": """
(atx_heading
  (inline) @name) @definition.section
(setext_heading
  (paragraph (inline) @name)) @definition.section
""",
    # Swift: class_declaration also covers struct / enum / actor.
    "swift": """
(class_declaration name: (type_identifier) @name) @definition.class
(protocol_declaration name: (type_identifier) @name) @definition.interface
(class_declaration
  (inheritance_specifier inherits_from: (user_type (type_identifier) @parent.class)))
(function_declaration name: (simple_identifier) @name) @definition.function
(protocol_function_declaration name: (simple_identifier) @name) @definition.method
(init_declaration "init" @name) @definition.method
(source_file
  (property_declaration
    name: (pattern bound_identifier: (simple_identifier) @name)) @definition.variable)
(class_body
  (property_declaration
    name: (pattern bound_identifier: (simple_identifier) @name)) @definition.variable)
(call_expression (simple_identifier) @ref.call)
(call_expression
  (navigation_expression
    suffix: (navigation_suffix suffix: (simple_identifier) @ref.call)))
(import_declaration (identifier) @import.module)
""",
    # Dart: a function body is a sibling of its signature, so definitions
    # span the signature only.
    "dart": """
(class_definition name: (identifier) @name) @definition.class
(class_definition superclass: (superclass (type_identifier) @parent.class))
(mixin_declaration (identifier) @name) @definition.class
(enum_declaration name: (identifier) @name) @definition.class
(extension_declaration name: (identifier) @name) @definition.class
(program (function_signature name: (identifier) @name) @definition.function)
(local_function_declaration
  (lambda_expression
    parameters: (function_signature name: (identifier) @name))) @definition.function
(program
  (static_final_declaration_list
    (static_final_declaration (identifier) @name) @definition.variable))
(method_signature (function_signature name: (identifier) @name)) @definition.method
(method_signature (getter_signature name: (identifier) @name)) @definition.method
(method_signature (setter_signature name: (identifier) @name)) @definition.method
(declaration (function_signature name: (identifier) @name)) @definition.method
(class_body
  (declaration
    (initialized_identifier_list
      (initialized_identifier (identifier) @name))) @definition.variable)
(_ (identifier) @ref.call . (selector (argument_part)))
(_
  (selector (unconditional_assignable_selector (identifier) @ref.call))
  .
  (selector (argument_part)))
(new_expression (type_identifier) @ref.new)
(library_import
  (import_specification (configurable_uri (uri (string_literal) @import.module))))
""",
    "lua": """
(function_declaration name: (identifier) @name) @definition.function
(function_declaration
  name: (dot_index_expression field: (identifier) @name)) @definition.method
(function_declaration
  name: (method_index_expression method: (identifier) @name)) @definition.method
(chunk
  (variable_declaration
    (assignment_statement
      (variable_list name: (identifier) @name))) @definition.variable)
(chunk
  (assignment_statement
    (variable_list name: (identifier) @name)) @definition.variable)
(function_call name: (identifier) @ref.call)
(function_call name: (dot_index_expression field: (identifier) @ref.call))
(function_call name: (method_index_expression method: (identifier) @ref.call))
((function_call
   name: (identifier) @_req
   arguments: (arguments (string content: (string_content) @import.module)))
 (#eq? @_req "require"))
""",
    "zig": """
(function_declaration name: (identifier) @name) @definition.function
(variable_declaration
  (identifier) @name
  [(struct_declaration) (enum_declaration) (union_declaration) (opaque_declaration)]) @definition.class
(source_file (variable_declaration (identifier) @name) @definition.variable)
(test_declaration (string (string_content) @name)) @definition.function
(call_expression function: (identifier) @ref.call)
(call_expression function: (field_expression member: (identifier) @ref.call))
((builtin_function
   (builtin_identifier) @_b
   (arguments (string (string_content) @import.module)))
 (#eq? @_b "@import"))
""",
    "haskell": """
(function name: (variable) @name) @definition.function
(bind name: (variable) @name) @definition.function
(data_type name: (name) @name) @definition.class
(newtype name: (name) @name) @definition.class
(type_synomym name: (name) @name) @definition.class
(class name: (name) @name) @definition.interface
(class_declarations (signature name: (variable) @name) @definition.method)
(apply function: (variable) @ref.call)
(apply function: (qualified id: (variable) @ref.call))
(import module: (module) @import.module)
(import_list (import_name (variable) @import.symbol))
(import_list (import_name (name) @import.symbol))
""",
    "ocaml": """
(compilation_unit
  (value_definition
    (let_binding pattern: (value_name) @name (parameter))) @definition.function)
(structure
  (value_definition
    (let_binding pattern: (value_name) @name (parameter))) @definition.function)
(compilation_unit
  (value_definition
    (let_binding pattern: (value_name) @name body: (fun_expression))) @definition.function)
(structure
  (value_definition
    (let_binding pattern: (value_name) @name body: (fun_expression))) @definition.function)
(compilation_unit
  (value_definition (let_binding pattern: (value_name) @name)) @definition.variable)
(structure
  (value_definition (let_binding pattern: (value_name) @name)) @definition.variable)
(type_definition (type_binding name: (type_constructor) @name)) @definition.class
(module_definition (module_binding (module_name) @name)) @definition.class
(module_type_definition (module_type_name) @name) @definition.interface
(class_definition (class_binding name: (class_name) @name)) @definition.class
(method_definition name: (method_name) @name) @definition.method
(application_expression function: (value_path (value_name) @ref.call))
(open_module module: (module_path) @import.module)
""",
    "ocaml_interface": """
(value_specification (value_name) @name) @definition.function
(type_definition (type_binding name: (type_constructor) @name)) @definition.class
(module_definition (module_binding (module_name) @name)) @definition.class
(module_type_definition (module_type_name) @name) @definition.interface
(open_module_signature module: (extended_module_path) @import.module)
""",
    # Julia: the definition signature is itself a call_expression, so calls
    # are matched per parent kind (never under `signature` / typed / where).
    "julia": """
(function_definition
  (signature (call_expression . (identifier) @name))) @definition.function
(function_definition
  (signature (typed_expression . (call_expression . (identifier) @name)))) @definition.function
(function_definition
  (signature (where_expression . (call_expression . (identifier) @name)))) @definition.function
(macro_definition
  (signature (call_expression . (identifier) @name))) @definition.function
(assignment . (call_expression . (identifier) @name)) @definition.function
(struct_definition (type_head (identifier) @name)) @definition.class
(struct_definition (type_head (binary_expression . (identifier) @name))) @definition.class
(struct_definition (type_head (binary_expression (identifier) @parent.class .)))
(abstract_definition (type_head (identifier) @name)) @definition.interface
(module_definition name: (identifier) @name) @definition.class
(source_file (const_statement (assignment . (identifier) @name)) @definition.variable)
(argument_list (call_expression . (identifier) @ref.call))
(return_statement (call_expression . (identifier) @ref.call))
(function_definition (call_expression . (identifier) @ref.call))
(macro_definition (call_expression . (identifier) @ref.call))
(module_definition (call_expression . (identifier) @ref.call))
(source_file (call_expression . (identifier) @ref.call))
(compound_statement (call_expression . (identifier) @ref.call))
(if_statement (call_expression . (identifier) @ref.call))
(elseif_clause (call_expression . (identifier) @ref.call))
(else_clause (call_expression . (identifier) @ref.call))
(for_statement (call_expression . (identifier) @ref.call))
(for_binding (call_expression . (identifier) @ref.call))
(while_statement (call_expression . (identifier) @ref.call))
(let_statement (call_expression . (identifier) @ref.call))
(do_clause (call_expression . (identifier) @ref.call))
(try_statement (call_expression . (identifier) @ref.call))
(catch_clause (call_expression . (identifier) @ref.call))
(finally_clause (call_expression . (identifier) @ref.call))
(binary_expression (call_expression . (identifier) @ref.call))
(unary_expression (call_expression . (identifier) @ref.call))
(ternary_expression (call_expression . (identifier) @ref.call))
(parenthesized_expression (call_expression . (identifier) @ref.call))
(tuple_expression (call_expression . (identifier) @ref.call))
(vector_expression (call_expression . (identifier) @ref.call))
(matrix_row (call_expression . (identifier) @ref.call))
(comprehension_expression (call_expression . (identifier) @ref.call))
(range_expression (call_expression . (identifier) @ref.call))
(index_expression (call_expression . (identifier) @ref.call))
(field_expression (call_expression . (identifier) @ref.call))
(splat_expression (call_expression . (identifier) @ref.call))
(named_argument (call_expression . (identifier) @ref.call))
(macro_argument_list (call_expression . (identifier) @ref.call))
(string_interpolation (call_expression . (identifier) @ref.call))
(open_tuple (call_expression . (identifier) @ref.call))
(arrow_function_expression (call_expression . (identifier) @ref.call))
(let_binding (call_expression . (identifier) @ref.call))
(assignment (operator) . (call_expression . (identifier) @ref.call))
(compound_assignment_expression (operator) . (call_expression . (identifier) @ref.call))
(call_expression . (field_expression (identifier) @ref.call .))
(broadcast_call_expression . (identifier) @ref.call)
(using_statement (identifier) @import.module)
(using_statement (scoped_identifier) @import.module)
(import_statement (identifier) @import.module)
(import_statement (scoped_identifier) @import.module)
(selected_import . (identifier) @import.module)
(selected_import . (scoped_identifier) @import.module)
(selected_import . (_) (identifier) @import.symbol)
""",
    "perl": """
(subroutine_declaration_statement name: (bareword) @name) @definition.function
(method_declaration_statement name: (bareword) @name) @definition.method
(package_statement name: (package) @name) @definition.class
(class_statement name: (package) @name) @definition.class
(function_call_expression function: (function) @ref.call)
(ambiguous_function_call_expression function: (function) @ref.call)
(method_call_expression method: (method) @ref.call)
(use_statement module: (package) @import.module)
(require_expression (bareword) @import.module)
""",
    "powershell": """
(function_statement (function_name) @name) @definition.function
(class_statement . (simple_name) @name) @definition.class
(class_statement (simple_name) (simple_name) @parent.class)
(enum_statement (simple_name) @name) @definition.class
(class_method_definition (simple_name) @name) @definition.method
(class_property_definition (variable) @name) @definition.variable
(command command_name: (command_name) @ref.call)
(invokation_expression (member_name (simple_name) @ref.call))
((command
   command_name: (command_name) @_c
   command_elements: (command_elements (generic_token) @import.module))
 (#match? @_c "^[Ii]mport-[Mm]odule$"))
((command
   command_name: (command_name) @_u
   command_elements: (command_elements (generic_token) @import.module .))
 (#eq? @_u "using"))
""",
    "objc": """
(class_interface . (identifier) @name) @definition.class
(class_interface superclass: (identifier) @parent.class)
(class_implementation . (identifier) @name) @definition.class
(protocol_declaration . (identifier) @name) @definition.interface
(protocol_declaration (method_declaration (identifier) @name) @definition.method)
(method_definition (identifier) @name) @definition.method
(function_definition
  declarator: (function_declarator declarator: (identifier) @name)) @definition.function
(property_declaration
  (struct_declaration (struct_declarator (identifier) @name))) @definition.variable
(call_expression function: (identifier) @ref.call)
(message_expression method: (identifier) @ref.call)
(preproc_include path: (_) @import.module)
""",
    "sql": """
(create_table (object_reference name: (identifier) @name)) @definition.class
(create_view (object_reference name: (identifier) @name)) @definition.class
(create_materialized_view (object_reference name: (identifier) @name)) @definition.class
(create_type (object_reference name: (identifier) @name)) @definition.class
(create_function (object_reference name: (identifier) @name)) @definition.function
(create_trigger (object_reference name: (identifier) @name)) @definition.function
(create_index column: (identifier) @name) @definition.variable
(create_sequence (object_reference name: (identifier) @name)) @definition.variable
(column_definition name: (identifier) @name) @definition.variable
(invocation (object_reference name: (identifier) @ref.call))
""",
    "toml": """
(table (bare_key) @name) @definition.class
(table (dotted_key) @name) @definition.class
(table (quoted_key) @name) @definition.class
(table_array_element (bare_key) @name) @definition.class
(table_array_element (dotted_key) @name) @definition.class
(document (pair (bare_key) @name) @definition.variable)
""",
    "xml": """
(document root: (element (STag (Name) @name)) @definition.class)
(document root: (element (EmptyElemTag (Name) @name)) @definition.class)
(document
  root: (element (content (element (STag (Name) @name)) @definition.variable)))
(document
  root: (element (content (element (EmptyElemTag (Name) @name)) @definition.variable)))
""",
    # Terraform / HCL: `resource "type" "name"` and `data` blocks are named
    # by their second label, one-label blocks by their label.
    "hcl": """
((block
   (identifier) @_t
   .
   (string_lit)
   .
   (string_lit (template_literal) @name)) @definition.class
 (#any-of? @_t "resource" "data"))
((block
   (identifier) @_t
   .
   (string_lit (template_literal) @name)
   .
   (block_start)) @definition.class
 (#any-of? @_t "module" "provider"))
((block
   (identifier) @_t
   .
   (string_lit (template_literal) @name)
   .
   (block_start)) @definition.variable
 (#any-of? @_t "variable" "output"))
((block
   (identifier) @_t
   (body (attribute (identifier) @name) @definition.variable))
 (#eq? @_t "locals"))
(function_call (identifier) @ref.call)
((block
   (identifier) @_t
   (body
     (attribute
       (identifier) @_s
       (expression (literal_value (string_lit (template_literal) @import.module))))))
 (#eq? @_t "module")
 (#eq? @_s "source"))
""",
    # Makefile: rule targets are functions, prerequisites are calls.
    "make": """
((rule (targets (word) @name)) @definition.function
 (#not-match? @name "^\\\\."))
(variable_assignment name: (word) @name) @definition.variable
(define_directive name: (word) @name) @definition.function
((rule
   (targets (word) @_t)
   normal: (prerequisites (word) @ref.call))
 (#not-match? @_t "^\\\\."))
(include_directive filenames: (list (word) @import.module))
""",
    # Svelte: <script> is raw text (no injection here); components used in
    # the markup and event handlers are the references.
    "svelte": """
((element (start_tag (tag_name) @ref.new)) (#match? @ref.new "^[A-Z]"))
((element (self_closing_tag (tag_name) @ref.new)) (#match? @ref.new "^[A-Z]"))
((attribute
   (attribute_name) @_a
   (expression (svelte_raw_text) @ref.call))
 (#match? @_a "^on:")
 (#match? @ref.call "^[A-Za-z_$][A-Za-z0-9_$]*$"))
""",
    "nix": """
(binding
  attrpath: (attrpath . (identifier) @name)
  expression: (function_expression)) @definition.function
(binding attrpath: (attrpath . (identifier) @name)) @definition.variable
(apply_expression function: (variable_expression name: (identifier) @ref.call))
(apply_expression
  function: (select_expression attrpath: (attrpath (identifier) @ref.call .)))
((apply_expression
   function: (variable_expression name: (identifier) @_i)
   argument: [(path_expression) (spath_expression)] @import.module)
 (#any-of? @_i "import" "callPackage"))
""",
    "groovy": """
(class_declaration name: (identifier) @name) @definition.class
(class_declaration superclass: (superclass (type_identifier) @parent.class))
(interface_declaration name: (identifier) @name) @definition.interface
(method_declaration name: (identifier) @name) @definition.method
(function_definition name: (identifier) @name) @definition.function
(field_declaration
  declarator: (variable_declarator name: (identifier) @name)) @definition.variable
(method_invocation name: (identifier) @ref.call)
(juxt_function_call name: (identifier) @ref.call)
(object_creation_expression type: (type_identifier) @ref.new)
(import_declaration (scoped_identifier) @import.module)
""",
    "fortran": """
(module (module_statement (name) @name)) @definition.class
(program (program_statement (name) @name)) @definition.class
(function (function_statement name: (name) @name)) @definition.function
(subroutine (subroutine_statement name: (name) @name)) @definition.function
(derived_type_definition (derived_type_statement (type_name) @name)) @definition.class
(interface (interface_statement (name) @name)) @definition.interface
(subroutine_call subroutine: (identifier) @ref.call)
(call_expression . (identifier) @ref.call)
(use_statement (module_name) @import.module)
""",
    "verilog": """
(module_declaration (module_header (simple_identifier) @name)) @definition.class
(interface_declaration (interface_ansi_header (interface_identifier) @name)) @definition.interface
(package_declaration (package_identifier) @name) @definition.class
(class_declaration (class_identifier) @name) @definition.class
(function_declaration
  (function_body_declaration (function_identifier) @name)) @definition.function
(task_declaration (task_body_declaration (task_identifier) @name)) @definition.function
(module_instantiation (simple_identifier) @ref.new)
(checker_instantiation (checker_identifier) @ref.new)
(tf_call (simple_identifier) @ref.call)
(function_subroutine_call (subroutine_call (tf_call (simple_identifier) @ref.call)))
(system_tf_call (system_tf_identifier) @ref.call)
(include_compiler_directive (double_quoted_string) @import.module)
(package_import_item (package_identifier) @import.module)
""",
    "vhdl": """
(entity_declaration entity: (identifier) @name) @definition.class
(architecture_definition architecture: (identifier) @name) @definition.class
(package_declaration package: (identifier) @name) @definition.class
(subprogram_definition
  (function_specification function: (_) @name)) @definition.function
(subprogram_definition
  (procedure_specification procedure: (_) @name)) @definition.function
(component_declaration component: (identifier) @name) @definition.interface
(name . (identifier) @ref.call . (parenthesis_group))
(instantiated_unit entity: (name) @ref.new)
(instantiated_unit component: (name) @ref.new)
(use_clause (selected_name_list (selected_name) @import.module))
""",
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
    column: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class FileParse:
    file: str
    language: str
    lines: int
    entities: list[Entity]
    edges: list[RawEdge]
    # File-level text chunks ({idx, line_start, line_end, body}) for
    # plain-text files, symbol-less grammar files and notebook markdown.
    chunks: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


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
    """Grammar language for a path (basename first, then extension), or
    None when no grammar claims it. Notebooks report "ipynb"."""
    name = path.name.lower()
    lang = FILENAME_TO_LANG.get(name)
    if lang:
        return lang
    for pre, lk in FILENAME_PREFIX_TO_LANG.items():
        if name.startswith(pre):
            return lk
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        return "ipynb"
    return EXT_TO_LANG.get(suffix)


def text_kind_for(path: Path) -> str | None:
    """Text kind from the name alone (no IO), or None when unknown."""
    name = path.name.lower()
    k = TEXT_NAME_KINDS.get(name)
    if k:
        return k
    for pre, kind in TEXT_NAME_PREFIX_KINDS.items():
        if name.startswith(pre):
            return kind
    return TEXT_EXT_KINDS.get(path.suffix.lower())


def looks_binary(head: bytes) -> bool:
    """NUL byte in the sniff window, or undecodable as UTF-8 / cp1252."""
    if not head:
        return False
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
        return False
    except UnicodeDecodeError as exc:
        # A multi-byte sequence cut by the sniff window is still text.
        if exc.start >= len(head) - 4:
            return False
    try:
        head.decode("cp1252")
        return False
    except UnicodeDecodeError:
        return True


def decode_text(data: bytes) -> str | None:
    """UTF-8 (BOM stripped), else cp1252 as a last resort; None if binary."""
    if b"\x00" in data[:SNIFF_BYTES]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return None


def classify_file(path: Path, sniff: bool = True, sniff_known: bool = True) -> str | None:
    """What the indexer does with a file: a grammar language key, "text:<kind>"
    for the plain-text fallback, or None (skip: binary / unreadable).

    `sniff=False` never reads the file (used for paths that no longer
    exist, e.g. a deleted file in the watcher)."""
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTS:
        return None
    lang = detect_language(path)
    if lang is not None:
        return lang
    kind = text_kind_for(path)
    if not sniff or (kind and not sniff_known):
        # a known text kind (README, .toml, .ini ...) is decoded at parse time
        # anyway; only unknown names need the binary sniff up front
        return f"text:{kind or 'text'}"
    try:
        with open(path, "rb") as fh:
            head = fh.read(SNIFF_BYTES)
    except OSError:
        return None
    if looks_binary(head):
        return None
    return f"text:{kind or 'text'}"


def is_indexable(path: Path) -> bool:
    return classify_file(path, sniff=path.exists()) is not None


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
    # Order matters for the dedup pass below: kinds listed earlier win when
    # the same span gets matched by multiple captures (e.g. JS/TS
    # `const foo = () => ...` matches both definition.function and
    # definition.variable). We prefer the richer kind.
    "definition.function": "function",
    "definition.method": "method",
    "definition.class": "class",
    "definition.interface": "interface",
    "definition.variable": "variable",
    "definition.section": "section",
}

_KIND_PRIORITY = {
    "class": 0,
    "interface": 0,
    "method": 1,
    "function": 1,
    "variable": 2,
    "section": 3,
}


_MEMBER_PARENTS = {
    "attribute": "object",                   # python
    "member_expression": "object",           # js / ts
    "selector_expression": "operand",        # go
    "field_expression": "value",             # rust
    "member_access_expression": "expression",  # c#
    "method_invocation": "object",           # java (name is a field of the call)
    "call": "receiver",                      # ruby
    "navigation_expression": None,           # kotlin
    "dot_index_expression": "table",         # lua  (a.b())
    "method_index_expression": "table",      # lua  (a:b())
    "value_path": None,                      # ocaml (List.iter)
    "method_call_expression": "invocant",    # perl  ($obj->m())
    "message_expression": "receiver",        # objc  ([obj m])
}
_SIMPLE_RECV = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*){0,3}$")


def _receiver_text(node: ts.Node, source: bytes) -> str | None:
    """For a call-name node, return the receiver text of a member call
    (`auth` in `auth.login()`), "" when the receiver is a complex
    expression, or None when the call is a bare name."""
    parent = node.parent
    if parent is None:
        return None
    field_name = _MEMBER_PARENTS.get(parent.type, "missing")
    if field_name == "missing":
        return None
    recv_node = parent.child_by_field_name(field_name) if field_name else None
    if recv_node is None:
        # method_invocation without an object / ruby call without receiver
        if parent.type in ("method_invocation", "call"):
            return None
        recv_node = parent.named_children[0] if parent.named_children else None
    if recv_node is None or recv_node.id == node.id:
        return None
    text = source[recv_node.start_byte:recv_node.end_byte].decode("utf-8", errors="replace")
    if len(text) > 80 or not _SIMPLE_RECV.match(text):
        return ""
    return text


def _import_module_for(sym_node: ts.Node, module_nodes: list[ts.Node],
                       source: bytes) -> str | None:
    """The module a symbol import belongs to: the @import.module capture
    inside the nearest ancestor import statement."""
    anc = sym_node.parent
    for _ in range(5):
        if anc is None:
            return None
        for m in module_nodes:
            if anc.start_byte <= m.start_byte and m.end_byte <= anc.end_byte:
                return source[m.start_byte:m.end_byte].decode(
                    "utf-8", errors="replace").strip("'\"<>`")
        anc = anc.parent
    return None


def _import_alias(sym_node: ts.Node, source: bytes) -> str | None:
    parent = sym_node.parent
    if parent is None:
        return None
    alias = parent.child_by_field_name("alias")
    if alias is None or alias.id == sym_node.id:
        return None
    return source[alias.start_byte:alias.end_byte].decode("utf-8", errors="replace").strip()


def _string_comment_lines(root: ts.Node) -> set[int]:
    """1-based line numbers covered by comments or multi-line strings."""
    out: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        t = node.type
        if "comment" in t or ("string" in t and node.start_point.row != node.end_point.row):
            for r in range(node.start_point.row, node.end_point.row + 1):
                out.add(r + 1)
            continue
        stack.extend(node.children)
    return out


_HTML_DROP = re.compile(r"<(script|style)\b.*?</\1\s*>", re.S | re.I)
_HTML_TAG = re.compile(r"<[^>]*>")


def _chunk_source_text(text: str, language: str) -> list[dict]:
    """File-level chunks for a symbol-less / plain-text file. HTML/XML-ish
    markup is reduced to its text per line so the line numbers still point
    at the original file."""
    from docgraph.summary import text_chunks
    if language in ("html",):
        # blank out script/style bodies but keep their newlines
        text = _HTML_DROP.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        text = "\n".join(_HTML_TAG.sub(" ", ln) for ln in text.split("\n"))
    return [{"idx": i, "line_start": s, "line_end": e, "body": b}
            for i, (s, e, b) in enumerate(text_chunks(text))]


def _rel(path: Path, repo_root: Path, rel_override: str | None) -> str:
    return rel_override if rel_override is not None else str(path.relative_to(repo_root)).replace("\\", "/")


def parse_file(path: Path, repo_root: Path, rel_override: str | None = None,
               text_fallback: bool = True) -> FileParse | None:
    """Parse one file. Grammar files give entities + edges; files no grammar
    claims (or whose grammar is not installed) fall back to plain text when
    they decode as text: a File with `language` = the detected kind and
    line/paragraph `chunks`. Grammar files without any function/class get
    the same file-level chunks so they stay searchable."""
    kind = classify_file(path)
    if kind is None:
        return None
    try:
        source = path.read_bytes()
    except OSError:
        return None
    rel = _rel(path, repo_root, rel_override)
    if kind == "ipynb":
        return _parse_notebook(source, rel, text_fallback=text_fallback)
    if kind.startswith("text:"):
        if not text_fallback:
            return None
        return _parse_text(source, rel, kind[5:])
    if _get_parser(kind) is None:
        # Grammar wheel missing / broken: plain-text path under the same key.
        if not text_fallback:
            return None
        return _parse_text(source, rel, kind)
    fp = _parse_grammar(source, kind, rel)
    if fp is None:
        return _parse_text(source, rel, kind) if text_fallback else None
    if text_fallback and not any(e.kind in ("function", "method", "class", "interface")
                                 for e in fp.entities):
        text = decode_text(source)
        if text is not None:
            fp.chunks = _chunk_source_text(text, kind)
    return fp


def _parse_text(source: bytes, rel: str, kind: str) -> FileParse | None:
    text = decode_text(source)
    if text is None:
        return None
    return FileParse(file=rel, language=kind or "text", lines=text.count("\n") + 1,
                     entities=[], edges=[], chunks=_chunk_source_text(text, kind))


# Jupyter kernel language names -> grammar keys.
_NB_LANG = {"python": "python", "python3": "python", "ipython": "python",
            "r": "r", "julia": "julia", "javascript": "javascript",
            "typescript": "typescript", "scala": "scala", "ruby": "ruby",
            "bash": "bash", "sh": "bash", "go": "go", "rust": "rust",
            "c++": "cpp", "cpp": "cpp", "java": "java", "kotlin": "kotlin",
            "csharp": "c_sharp", "c#": "c_sharp", "lua": "lua", "haskell": "haskell",
            "sql": "sql", "powershell": "powershell"}


def notebook_text(data: bytes) -> tuple[str, str, list[tuple[str, int, int]]] | None:
    """(virtual_text, language, [(cell_type, line_start, line_end)]) for an
    .ipynb: every cell's source in order, one blank line between cells. The
    virtual line numbers are what entities / chunks of a notebook refer to
    (the UI shows the same text for the file)."""
    import json as _json
    raw = decode_text(data)
    if raw is None:
        return None
    try:
        nb = _json.loads(raw)
    except ValueError:
        return None
    if not isinstance(nb, dict):
        return None
    meta = nb.get("metadata") or {}
    lang = ""
    if isinstance(meta, dict):
        li = meta.get("language_info") or {}
        ks = meta.get("kernelspec") or {}
        lang = str((li.get("name") if isinstance(li, dict) else "") or
                   (ks.get("language") if isinstance(ks, dict) else "") or "python").lower()
    lang_key = _NB_LANG.get(lang, lang or "python")
    parts: list[str] = []
    spans: list[tuple[str, int, int]] = []
    line = 1
    for cell in nb.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        src = cell.get("source") or ""
        if isinstance(src, list):
            src = "".join(str(s) for s in src)
        src = str(src).rstrip("\n")
        n = src.count("\n") + 1
        spans.append((str(cell.get("cell_type") or "code"), line, line + n - 1))
        parts.append(src)
        line += n + 1
    return "\n\n".join(parts) + ("\n" if parts else ""), lang_key, spans


def _parse_notebook(source: bytes, rel: str, text_fallback: bool = True) -> FileParse | None:
    """Code cells parse as the notebook's language (markdown / raw cells are
    blanked to keep line numbers); markdown cells become text chunks."""
    nt = notebook_text(source)
    if nt is None:
        return _parse_text(source, rel, "json") if text_fallback else None
    text, lang_key, spans = nt
    lines = text.split("\n")
    code_lines = list(lines)
    md_blocks: list[tuple[int, int]] = []
    for ctype, s, e in spans:
        if ctype != "code":
            for i in range(s - 1, min(e, len(code_lines))):
                code_lines[i] = ""
            if ctype == "markdown":
                md_blocks.append((s, e))
    fp: FileParse | None = None
    if _get_parser(lang_key) is not None:
        fp = _parse_grammar("\n".join(code_lines).encode("utf-8"), lang_key, rel)
    if fp is None:
        fp = FileParse(file=rel, language=lang_key, lines=len(lines), entities=[], edges=[])
    fp.lines = len(lines)
    chunks: list[dict] = []
    from docgraph.summary import text_chunks
    for s, e in md_blocks:
        block = "\n".join(lines[s - 1:e])
        for cs, ce, body in text_chunks(block):
            chunks.append({"idx": len(chunks), "line_start": s + cs - 1,
                           "line_end": s + ce - 1, "body": body})
    if _get_parser(lang_key) is None:
        # No grammar for the kernel language: code cells are text chunks too.
        for ctype, s, e in spans:
            if ctype == "code":
                block = "\n".join(lines[s - 1:e])
                for cs, ce, body in text_chunks(block):
                    chunks.append({"idx": len(chunks), "line_start": s + cs - 1,
                                   "line_end": s + ce - 1, "body": body})
    fp.chunks = chunks if text_fallback else []
    fp.extra = {"notebook": True}
    return fp


def _parse_grammar(source: bytes, lang_key: str, rel: str) -> FileParse | None:
    parser = _get_parser(lang_key)
    if parser is None:
        return None
    try:
        tree = parser.parse(source)
    except Exception:
        return None
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

    # Dedup by (name, line_start, line_end). Some node ranges are matched by
    # multiple capture patterns — e.g. JS/TS `const foo = () => ...` matches
    # both `definition.function` and `definition.variable`. Keep the higher-
    # priority kind and drop the rest. Without this, the same identifier
    # would land in the graph twice with the same qname.
    if entities:
        keepers: dict[tuple[str, int, int], int] = {}
        for i, e in enumerate(entities):
            key = (e.name, e.line_start, e.line_end)
            cur = keepers.get(key)
            if cur is None:
                keepers[key] = i
            else:
                if _KIND_PRIORITY.get(e.kind, 99) < _KIND_PRIORITY.get(entities[cur].kind, 99):
                    keepers[key] = i
        kept_idxs = set(keepers.values())
        if len(kept_idxs) != len(entities):
            entities = [entities[i] for i in sorted(kept_idxs)]
            # Rebuild defs in lockstep so downstream resolution stays aligned.
            keep_qnames = {(e.qname, e.line_start) for e in entities}
            defs = [
                (n, q, k) for (n, q, k) in defs
                if (q, n.start_point.row + 1) in keep_qnames
            ]

    # Re-scope methods inside classes. Key on node identity, not on the
    # original qname string — two methods of the same name in different
    # classes (e.g. Square.area and Circle.area) share the base qname
    # `file::area` and the old per-qname remap collapsed both into one.
    # entities[i] corresponds to defs[i] after the dedup pass above; the
    # zip below relies on that ordering.
    node_to_qname: dict[int, str] = {}
    any_remap = False
    for d_node, qname, kind in defs:
        if kind in ("class", "interface"):
            continue
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
            node_to_qname[id(d_node)] = f"{enclosing_class[0]}::{qname.split('::')[-1]}"
            any_remap = True

    if any_remap:
        for i, (d_node, _q, _k) in enumerate(defs):
            new_q = node_to_qname.get(id(d_node))
            if new_q and i < len(entities):
                entities[i].qname = new_q
        defs = [(n, node_to_qname.get(id(n), q), k) for (n, q, k) in defs]

    # Edges: refs. For member calls (`auth.login()`, `this.x()`) we keep the
    # receiver text -- the resolution cascade (resolve.py) uses it to pick
    # between same-named candidates.
    for ref_cap, edge_kind in [("ref.call", "CALLS"), ("ref.new", "INSTANTIATES")]:
        for r_node in caps.get(ref_cap, []):
            target = source[r_node.start_byte:r_node.end_byte].decode("utf-8", errors="replace")
            enc = _enclosing(r_node, defs)
            extra: dict = {}
            recv = _receiver_text(r_node, source)
            if recv is not None:
                extra["attr"] = True
                if recv:
                    extra["recv"] = recv
            raw_edges.append(RawEdge(
                kind=edge_kind,
                src_file=rel,
                src_qname=enc[0] if enc else None,
                target_name=target,
                line=r_node.start_point.row + 1,
                column=r_node.start_point.column + 1,
                extra=extra,
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
            line=i_node.start_point.row + 1,
            column=i_node.start_point.column + 1,
        ))

    # Symbol-level imports — `from x import Y` (Python), `import {Y} from "x"` (JS/TS).
    # We don't try to associate each symbol back to a specific module here;
    # the index resolver matches the symbol name against any imported file's
    # exported entities, falling back to a global lookup. Java's qualified
    # imports (`import a.b.C;`) already terminate in the symbol name, so the
    # last component of @import.module is the symbol — handled in index.py.
    module_nodes = caps.get("import.module", [])
    for s_node in caps.get("import.symbol", []):
        sym = source[s_node.start_byte:s_node.end_byte].decode("utf-8", errors="replace").strip()
        if not sym:
            continue
        extra = {}
        mod = _import_module_for(s_node, module_nodes, source)
        if mod:
            extra["module"] = mod
        alias = _import_alias(s_node, source)
        if alias and alias != sym:
            extra["alias"] = alias
        raw_edges.append(RawEdge(
            kind="IMPORTS_SYMBOL",
            src_file=rel,
            src_qname=None,
            target_name=sym,
            line=s_node.start_point.row + 1,
            column=s_node.start_point.column + 1,
            extra=extra,
        ))

    # Framework routes + MCP tools (table-driven regexes, see frameworks.py).
    # Hits inside docstrings / comments are dropped via the syntax tree.
    try:
        from docgraph.frameworks import detect as _detect_frameworks
        text = source.decode("utf-8", errors="replace")
        ent_defs = [(e.kind, e.qname, e.line_start, e.line_end) for e in entities]
        hits = _detect_frameworks(text, lang_key, ent_defs)
        if hits:
            skip = _string_comment_lines(tree.root_node)
            for h in hits:
                if h.line in skip:
                    continue
                raw_edges.append(RawEdge(
                    kind=h.kind,
                    src_file=rel,
                    src_qname=h.handler_qname,
                    target_name=h.handler_name,
                    line=h.line,
                    extra={
                        "name": h.name, "framework": h.framework,
                        "method": h.method, "path": h.path,
                        "tool_kind": h.tool_kind, **h.extra,
                    },
                ))
    except Exception as exc:  # noqa: BLE001 - detection must never fail a parse
        log.debug("framework detection failed for %s: %s", rel, exc)

    return FileParse(
        file=rel,
        language=lang_key,
        lines=lines,
        entities=entities,
        edges=raw_edges,
    )


def parse_worker(args: tuple) -> dict | None:
    """ProcessPool entry point (kept in this module so a spawned worker
    imports tree-sitter only, not the indexer / DB / CLI stack).
    args = (file_path, repo_root, logical_rel[, text_fallback])."""
    from dataclasses import asdict
    file_path, repo_root, rel_override = args[:3]
    text_fallback = bool(args[3]) if len(args) > 3 else True
    try:
        fp = parse_file(Path(file_path), Path(repo_root), rel_override=rel_override,
                        text_fallback=text_fallback)
        if fp is None:
            return None
        return {
            "file": fp.file,
            "language": fp.language,
            "lines": fp.lines,
            "entities": [asdict(e) for e in fp.entities],
            "edges": [asdict(e) for e in fp.edges],
            "chunks": fp.chunks,
            "extra": fp.extra,
        }
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{file_path}: {e}"}
