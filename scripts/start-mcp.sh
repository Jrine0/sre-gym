#!/bin/bash
# Start the SRE Gym MCP server for Claude Code integration
#
# This launches the MCP server that exposes kubectl tools as Claude Code tools.
# The server uses JSON-RPC 2.0 over stdio for communication.
#
# Usage:
#   ./scripts/start-mcp.sh
#
# Or with Docker:
#   docker run --rm -it sre-gym python -m sre_gym.mcp_server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Warning: kubectl not found in PATH. Some MCP tools may not work."
fi

# Check if kind cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "Warning: No Kubernetes cluster accessible. Start a kind cluster first:"
    echo "  kind create cluster --name sre-gym"
fi

echo "Starting SRE Gym MCP server..."
exec python -m sre_gym.mcp_server
