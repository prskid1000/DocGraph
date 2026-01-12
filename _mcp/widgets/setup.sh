#!/bin/bash
# Setup script for DocGraph widgets - installs Node.js dependencies (Linux/macOS)

set -e

echo "Installing widget dependencies..."

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install Node.js first."
    echo "   Visit https://nodejs.org/ to install Node.js"
    exit 1
fi

echo "Found npm: $(npm --version)"
echo "Found node: $(node --version)"

npm install

if [ $? -eq 0 ]; then
    echo "Widget dependencies installed!"
    echo ""
    echo "To build widgets manually, run: npm run build"
    echo "Or widgets will be built automatically on MCP server startup."
else
    echo "Failed to install dependencies"
    exit 1
fi
