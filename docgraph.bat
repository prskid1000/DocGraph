@echo off
REM docgraph CLI shim - forwards to the .venv next to this script.
"%~dp0.venv\Scripts\docgraph.exe" %*
