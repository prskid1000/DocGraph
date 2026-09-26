"""Tree-sitter grammars beyond the core set: one small fixture per language.

Each case writes the fixture to tmp_path, runs `parse_file`, and checks the
definitions (name + kind) plus the CALLS / IMPORTS edges the tags query is
meant to produce. No index, no embedding model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import tree_sitter as ts

from docgraph import parse as P


@dataclass
class Case:
    filename: str
    lang: str
    source: str
    defs: set[tuple[str, str]]
    calls: set[str] = field(default_factory=set)
    imports: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    news: set[str] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)


CASES: list[Case] = [
    Case(
        "shapes.swift", "swift",
        """import Foundation

protocol Shape {
    func area() -> Double
}

class Circle: Shape {
    var radius: Double = 1.0
    func area() -> Double {
        return compute(radius)
    }
}

struct Point {
    let x: Int
}

func compute(_ r: Double) -> Double {
    let c = Circle()
    print(c.area())
    return r * r
}
""",
        defs={("Shape", "interface"), ("Circle", "class"), ("Point", "class"),
              ("compute", "function"), ("radius", "variable")},
        calls={"compute", "print", "area"},
        imports={"Foundation"},
        parents={"Shape"},
    ),
    Case(
        "shapes.dart", "dart",
        """import 'package:flutter/material.dart';

abstract class Shape {
  double area();
}

class Circle extends Shape {
  double radius = 1.0;
  double area() {
    return compute(radius);
  }
}

double compute(double r) {
  var c = Circle();
  print(c.area());
  return r * r;
}
""",
        defs={("Shape", "class"), ("Circle", "class"), ("area", "method"),
              ("compute", "function"), ("radius", "variable")},
        calls={"compute", "print", "area"},
        imports={"package:flutter/material.dart"},
        parents={"Shape"},
    ),
    Case(
        "mod.lua", "lua",
        """local json = require("dkjson")

local function helper(x)
  return x * 2
end

function M.greet(name)
  print(helper(1))
  return string.format("hi %s", name)
end

function top()
  M.greet("a")
end
""",
        defs={("helper", "function"), ("greet", "method"), ("top", "function"),
              ("json", "variable")},
        calls={"helper", "print", "format", "greet"},
        imports={"dkjson"},
    ),
    Case(
        "main.zig", "zig",
        """const std = @import("std");

const Point = struct {
    x: i32,
    pub fn len(self: Point) i32 {
        return helper(self.x);
    }
};

fn helper(x: i32) i32 {
    return x * 2;
}

pub fn main() void {
    std.debug.print("{}", .{helper(1)});
}
""",
        defs={("Point", "class"), ("len", "function"), ("helper", "function"),
              ("main", "function"), ("std", "variable")},
        calls={"helper", "print"},
        imports={"std"},
    ),
    Case(
        "Main.hs", "haskell",
        """module Main where

import Data.List (sortBy)
import qualified Data.Map as M

data Shape = Circle Double | Square Double

class Area a where
  area :: a -> Double

helper :: Int -> Int
helper x = x * 2

main :: IO ()
main = print (helper 1)
""",
        defs={("Shape", "class"), ("Area", "interface"), ("area", "method"),
              ("helper", "function"), ("main", "function")},
        calls={"print", "helper"},
        imports={"Data.List", "Data.Map"},
        symbols={"sortBy"},
    ),
    Case(
        "geo.ml", "ocaml",
        """open Printf

type shape = Circle of float | Square of float

module Geometry = struct
  let area s = match s with
    | Circle r -> 3.14 *. r *. r
    | Square s -> s *. s
end

let version = "1.0"

let helper x = x * 2

let main () =
  printf "%d" (helper 1);
  List.iter print_endline ["a"]
""",
        defs={("shape", "class"), ("Geometry", "class"), ("area", "function"),
              ("helper", "function"), ("main", "function"), ("version", "variable")},
        calls={"printf", "helper", "iter"},
        imports={"Printf"},
    ),
    Case(
        "geo.mli", "ocaml_interface",
        """open Stdlib

type shape

val area : shape -> float

module Geometry : sig
  val helper : int -> int
end
""",
        defs={("shape", "class"), ("area", "function")},
        imports={"Stdlib"},
    ),
    Case(
        "Geo.jl", "julia",
        """module Geo

using LinearAlgebra
import Base: show

abstract type Shape end

struct Circle <: Shape
    r::Float64
end

function area(c::Circle)
    return helper(c.r)
end

helper(x) = x * x

function main()
    println(area(Circle(1.0)))
end

end
""",
        defs={("Geo", "class"), ("Shape", "interface"), ("Circle", "class"),
              ("area", "function"), ("helper", "function"), ("main", "function")},
        calls={"helper", "println", "area", "Circle"},
        imports={"LinearAlgebra", "Base"},
        symbols={"show"},
        parents={"Shape"},
    ),
    Case(
        "Circle.pm", "perl",
        """package Geo::Circle;
use strict;
use List::Util qw(sum);

sub new {
    my ($class, %args) = @_;
    return bless {%args}, $class;
}

sub area {
    my $self = shift;
    return helper($self->{r}) + $self->scale();
}

sub helper {
    my $x = shift;
    print sum(1, 2);
    return $x * $x;
}

1;
""",
        defs={("Geo::Circle", "class"), ("new", "function"), ("area", "function"),
              ("helper", "function")},
        calls={"helper", "scale", "sum", "bless"},
        imports={"strict", "List::Util"},
    ),
    Case(
        "tools.ps1", "powershell",
        """Import-Module ActiveDirectory

class Circle {
    [double]$Radius
    [double] Area() {
        return $this.Radius * $this.Radius
    }
}

function Get-Helper {
    param([int]$x)
    Write-Output ($x * 2)
}

function Invoke-Main {
    Get-Helper -x 1
    $c = [Circle]::new()
    $c.Area()
}
""",
        defs={("Circle", "class"), ("Area", "method"), ("Get-Helper", "function"),
              ("Invoke-Main", "function")},
        calls={"Get-Helper", "Write-Output", "Area"},
        imports={"ActiveDirectory"},
    ),
    Case(
        "Circle.m", "objc",
        """#import <Foundation/Foundation.h>
#include "shape.h"

@protocol Shape
- (double)area;
@end

@interface Circle : NSObject <Shape>
@property double radius;
- (double)area;
@end

@implementation Circle
- (double)area {
    return helper(self.radius);
}
@end

double helper(double r) {
    NSLog(@"x");
    [obj doSomething];
    return r * r;
}
""",
        defs={("Shape", "interface"), ("Circle", "class"), ("area", "method"),
              ("helper", "function"), ("radius", "variable")},
        calls={"helper", "NSLog", "doSomething"},
        imports={"Foundation/Foundation.h", "shape.h"},
        parents={"NSObject"},
    ),
    Case(
        "schema.sql", "sql",
        """CREATE TABLE users (
    id INT PRIMARY KEY,
    name TEXT
);

CREATE VIEW active_users AS SELECT * FROM users WHERE id > 0;

CREATE FUNCTION add_one(x INT) RETURNS INT AS $$ SELECT x + 1 $$ LANGUAGE sql;

CREATE INDEX idx_users_name ON users (name);

SELECT count(*) FROM users;
""",
        defs={("users", "class"), ("active_users", "class"), ("add_one", "function"),
              ("idx_users_name", "variable"), ("id", "variable")},
        calls={"count"},
    ),
    Case(
        "pyproject.toml", "toml",
        """title = "demo"

[project]
name = "docgraph"
version = "1.0"

[tool.pytest.ini_options]
addopts = "-q"

[[bin]]
name = "tool"
""",
        defs={("project", "class"), ("tool.pytest.ini_options", "class"),
              ("bin", "class"), ("title", "variable")},
    ),
    Case(
        "pom.xml", "xml",
        """<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <groupId>com.example</groupId>
  <build>
    <plugin id="compiler"/>
  </build>
</project>
""",
        defs={("project", "class"), ("groupId", "variable"), ("build", "variable")},
    ),
    Case(
        "main.tf", "hcl",
        """terraform {
  required_version = ">= 1.0"
}

provider "aws" {
  region = "us-east-1"
}

variable "region" {
  type = string
}

resource "aws_s3_bucket" "logs" {
  bucket = "my-logs"
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
}

locals {
  name = upper("x")
}
""",
        defs={("aws", "class"), ("region", "variable"), ("logs", "class"),
              ("vpc", "class"), ("name", "variable")},
        calls={"upper"},
        imports={"terraform-aws-modules/vpc/aws"},
    ),
    Case(
        "Makefile", "make",
        "CC = gcc\ninclude common.mk\n\nall: build test\n\nbuild: main.o\n"
        "\t$(CC) -o app main.o\n\ntest:\n\t./run_tests.sh\n\n.PHONY: all build test\n",
        defs={("all", "function"), ("build", "function"), ("test", "function"),
              ("CC", "variable")},
        calls={"build", "test"},
        imports={"common.mk"},
    ),
    Case(
        "Counter.svelte", "svelte",
        """<script>
  import Button from './Button.svelte';
  let count = 0;
  function increment() {
    count += 1;
  }
</script>

<Button on:click={increment}>Clicked {count}</Button>
""",
        defs=set(),
        calls={"increment"},
        news={"Button"},
    ),
    Case(
        "shell.nix", "nix",
        """{ pkgs ? import <nixpkgs> {} }:

let
  helper = x: x + 1;
  version = "1.0";
in
pkgs.mkShell {
  buildInputs = [ pkgs.python3 ];
  shellHook = builtins.toString (helper 1);
}
""",
        defs={("helper", "function"), ("version", "variable"), ("shellHook", "variable")},
        calls={"helper", "mkShell", "toString"},
        imports={"nixpkgs"},
    ),
    Case(
        "Circle.groovy", "groovy",
        """import groovy.json.JsonSlurper

class Circle extends Shape {
    double radius = 1.0;
    double area() {
        return helper(radius)
    }
}

def helper(double x) {
    println "hi"
    return x * x
}
""",
        defs={("Circle", "class"), ("area", "method"), ("helper", "function"),
              ("radius", "variable")},
        calls={"helper", "println"},
        imports={"groovy.json.JsonSlurper"},
        parents={"Shape"},
    ),
    Case(
        "geometry.f90", "fortran",
        """module geometry
  use iso_fortran_env
  implicit none
contains
  function area(r) result(a)
    real :: r, a
    a = helper(r)
  end function area

  subroutine report(r)
    real :: r
    call print_it(r)
  end subroutine report
end module geometry

program main
  use geometry
  call report(1.0)
end program main
""",
        defs={("geometry", "class"), ("main", "class"), ("area", "function"),
              ("report", "function")},
        calls={"helper", "print_it", "report"},
        imports={"iso_fortran_env", "geometry"},
    ),
    Case(
        "top.v", "verilog",
        """`include "defs.vh"

module counter (input clk, output reg [3:0] q);
  always @(posedge clk) q <= q + 1;
endmodule

module top;
  wire clk;
  counter u1 (.clk(clk));
  function integer add(input integer a);
    add = a + inc(1);
  endfunction
  task show;
    $display("x");
  endtask
endmodule
""",
        defs={("counter", "class"), ("top", "class"), ("add", "function"),
              ("show", "function")},
        calls={"inc", "$display"},
        imports={"defs.vh"},
        news={"counter"},
    ),
    Case(
        "counter.vhd", "vhdl",
        """library ieee;
use ieee.std_logic_1164.all;

entity counter is
  port (clk : in std_logic);
end entity counter;

architecture rtl of counter is
  function add_one(a : integer) return integer is
  begin
    return a + 1;
  end function;
begin
  u1: entity work.adder port map (a => clk);
  process(clk)
    variable v : integer;
  begin
    v := add_one(v);
    log_it(v);
  end process;
end architecture rtl;

package utils is
end package utils;
""",
        defs={("counter", "class"), ("rtl", "class"), ("utils", "class"),
              ("add_one", "function")},
        calls={"add_one", "log_it"},
        imports={"ieee.std_logic_1164.all"},
        news={"adder"},
    ),
]


def _parse(tmp_path: Path, case: Case) -> P.FileParse:
    path = tmp_path / case.filename
    path.write_text(case.source, encoding="utf-8")
    fp = P.parse_file(path, tmp_path)
    assert fp is not None, f"{case.lang}: parse_file returned None (grammar not loaded?)"
    return fp


def _targets(fp: P.FileParse, kind: str) -> set[str]:
    return {e.target_name for e in fp.edges if e.kind == kind}


@pytest.mark.parametrize("case", CASES, ids=[c.lang for c in CASES])
def test_language_fixture(tmp_path: Path, case: Case) -> None:
    fp = _parse(tmp_path, case)
    assert fp.language == case.lang

    got = {(e.name, e.kind) for e in fp.entities}
    missing = case.defs - got
    assert not missing, f"{case.lang}: missing definitions {missing}; got {sorted(got)}"

    calls = _targets(fp, "CALLS")
    assert case.calls <= calls, f"{case.lang}: calls {case.calls - calls} missing; got {calls}"
    imports = _targets(fp, "IMPORTS")
    assert case.imports <= imports, f"{case.lang}: imports {case.imports - imports}; got {imports}"
    symbols = _targets(fp, "IMPORTS_SYMBOL")
    assert case.symbols <= symbols, f"{case.lang}: symbols {case.symbols - symbols}"
    news = _targets(fp, "INSTANTIATES")
    assert case.news <= news, f"{case.lang}: instantiations {case.news - news}"
    parents = _targets(fp, "INHERITS")
    assert case.parents <= parents, f"{case.lang}: parents {case.parents - parents}"


@pytest.mark.parametrize("case", CASES, ids=[c.lang for c in CASES])
def test_language_fixture_parses_cleanly(case: Case) -> None:
    parser = P._get_parser(case.lang)
    assert parser is not None
    tree = parser.parse(case.source.encode("utf-8"))
    assert not tree.root_node.has_error, f"{case.lang}: fixture has syntax errors"


_NEW_LANGS = sorted({c.lang for c in CASES})


@pytest.mark.parametrize("lang", _NEW_LANGS)
def test_tags_query_compiles(lang: str) -> None:
    language = P._load_language(lang)
    assert language is not None, f"{lang}: grammar failed to load"
    src = P.TAGS_QUERIES[lang]
    assert src.strip(), f"{lang}: empty tags query"
    ts.Query(language, src)  # raises on a bad pattern / node type


@pytest.mark.parametrize("name,lang", [
    ("Makefile", "make"),
    ("GNUmakefile", "make"),
    ("Makefile.win", "make"),
    ("rules.mk", "make"),
    ("Jenkinsfile", "groovy"),
    ("build.gradle", "groovy"),
    ("Cargo.lock", "toml"),
    ("App.csproj", "xml"),
    ("vars.tfvars", "hcl"),
    ("Mod.psm1", "powershell"),
    ("iface.mli", "ocaml_interface"),
    ("core.sv", "verilog"),
])
def test_filename_detection(name: str, lang: str) -> None:
    assert P.detect_language(Path(name)) == lang
    assert P.classify_file(Path(name), sniff=False) == lang
