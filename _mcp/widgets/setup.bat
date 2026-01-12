@echo off
REM Setup script for DocGraph widgets - installs Node.js dependencies (Windows)

echo Installing widget dependencies...

REM Check if npm is available
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo npm is not installed. Please install Node.js first.
    echo    Visit https://nodejs.org/ to install Node.js
    exit /b 1
)

echo Found npm
npm --version
echo Found node
node --version

call npm install

if %errorlevel% equ 0 (
    echo Widget dependencies installed!
    echo.
    echo To build widgets manually, run: npm run build
    echo Or widgets will be built automatically on MCP server startup.
) else (
    echo Failed to install dependencies
    exit /b 1
)
