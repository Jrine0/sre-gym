"""Model Context Protocol (MCP) server exposing SRE Gym actions as tools.

This allows Claude Code and other MCP-compatible IDEs to interact with the gym
using the standard tool-call interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import json


@dataclass
class MCPTool:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict]


TOOLS: list[MCPTool] = []


def register_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
):
    """Decorator to register a function as an MCP tool."""
    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        TOOLS.append(MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=func,
        ))
        return func
    return decorator


# =============================================================================
# Kubernetes Diagnostic Tools
# =============================================================================

@register_tool(
    name="kubectl_get_pods",
    description="List all pods in a namespace with status and restarts",
    input_schema={
        "type": "object",
        "properties": {
            "namespace": {"type": "string", "default": "default"},
            "label_selector": {"type": "string", "default": ""},
        },
    },
)
def kubectl_get_pods(namespace: str = "default", label_selector: str = "") -> dict:
    """Get pods - wrapper for kubectl get pods."""
    import subprocess
    args = ["kubectl", "get", "pods", "-n", namespace, "-o", "wide"]
    if label_selector:
        args.extend(["-l", label_selector])
    result = subprocess.run(args, capture_output=True, text=True)
    return {"output": result.stdout, "error": result.stderr}


@register_tool(
    name="kubectl_describe_pod",
    description="Get detailed information about a specific pod including events",
    input_schema={
        "type": "object",
        "required": ["pod_name"],
        "properties": {
            "pod_name": {"type": "string"},
            "namespace": {"type": "string", "default": "default"},
        },
    },
)
def kubectl_describe_pod(pod_name: str, namespace: str = "default") -> dict:
    """Describe pod - shows events and why a pod might be failing."""
    import subprocess
    result = subprocess.run(
        ["kubectl", "describe", "pod", pod_name, "-n", namespace],
        capture_output=True,
        text=True,
    )
    return {"output": result.stdout, "error": result.stderr}


@register_tool(
    name="kubectl_get_events",
    description="Get recent Kubernetes events sorted by time",
    input_schema={
        "type": "object",
        "properties": {
            "namespace": {"type": "string", "default": "default"},
        },
    },
)
def kubectl_get_events(namespace: str = "default") -> dict:
    """Get recent events."""
    import subprocess
    result = subprocess.run(
        ["kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
        capture_output=True,
        text=True,
    )
    return {"output": result.stdout, "error": result.stderr}


@register_tool(
    name="kubectl_apply",
    description="Apply a Kubernetes manifest (YAML/JSON)",
    input_schema={
        "type": "object",
        "required": ["manifest"],
        "properties": {
            "manifest": {"type": "string", "description": "YAML manifest content"},
            "namespace": {"type": "string", "default": "default"},
        },
    },
)
def kubectl_apply(manifest: str, namespace: str = "default") -> dict:
    """Apply a manifest - creates or updates resources."""
    import subprocess
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        capture_output=True,
        text=True,
    )
    return {"output": result.stdout, "error": result.stderr}


@register_tool(
    name="kubectl_delete",
    description="Delete a Kubernetes resource",
    input_schema={
        "type": "object",
        "required": ["resource_kind", "resource_name"],
        "properties": {
            "resource_kind": {"type": "string", "enum": ["pod", "deployment", "configmap", "service"]},
            "resource_name": {"type": "string"},
            "namespace": {"type": "string", "default": "default"},
        },
    },
)
def kubectl_delete(resource_kind: str, resource_name: str, namespace: str = "default") -> dict:
    """Delete a resource."""
    import subprocess
    result = subprocess.run(
        ["kubectl", "delete", resource_kind, resource_name, "-n", namespace],
        capture_output=True,
        text=True,
    )
    return {"output": result.stdout, "error": result.stderr}


@register_tool(
    name="kubectl_logs",
    description="Get logs from a pod container",
    input_schema={
        "type": "object",
        "required": ["pod_name"],
        "properties": {
            "pod_name": {"type": "string"},
            "namespace": {"type": "string", "default": "default"},
            "container": {"type": "string", "default": ""},
            "tail": {"type": "integer", "default": 50},
        },
    },
)
def kubectl_logs(pod_name: str, namespace: str = "default", container: str = "", tail: int = 50) -> dict:
    """Get pod logs."""
    import subprocess
    args = ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={tail}"]
    if container:
        args.extend(["-c", container])
    result = subprocess.run(args, capture_output=True, text=True)
    return {"output": result.stdout, "error": result.stderr}


@register_tool(
    name="kubectl_patch",
    description="Patch a Kubernetes resource (JSON merge patch)",
    input_schema={
        "type": "object",
        "required": ["resource_kind", "resource_name", "patch"],
        "properties": {
            "resource_kind": {"type": "string"},
            "resource_name": {"type": "string"},
            "patch": {"type": "string", "description": "JSON patch content"},
            "namespace": {"type": "string", "default": "default"},
        },
    },
)
def kubectl_patch(resource_kind: str, resource_name: str, patch: str, namespace: str = "default") -> dict:
    """Patch a resource."""
    import subprocess
    result = subprocess.run(
        ["kubectl", "patch", resource_kind, resource_name, "-n", namespace, "--type=merge", "-p", patch],
        capture_output=True,
        text=True,
    )
    return {"output": result.stdout, "error": result.stderr}


def get_mcp_tools() -> list[dict]:
    """Return all registered MCP tools as JSON schema definitions."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }
        for tool in TOOLS
    ]


def handle_mcp_call(tool_name: str, arguments: dict) -> dict:
    """Handle an MCP tool call, routing to the correct handler."""
    for tool in TOOLS:
        if tool.name == tool_name:
            return tool.handler(**arguments)
    return {"error": f"Unknown tool: {tool_name}"}


# =============================================================================
# MCP Server (JSON-RPC 2.0 over stdio)
# =============================================================================

def serve():
    """Run MCP server on stdin/stdout (JSON-RPC 2.0 protocol)."""
    import sys
    import json

    for line in sys.stdin:
        if not line.strip():
            continue

        request = json.loads(line)
        method = request.get("method")
        msg_id = request.get("id")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "sre-gym", "version": "0.1.0"},
                },
            }
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": get_mcp_tools()},
            }
        elif method == "tools/call":
            tool_name = request["params"]["name"]
            args = request["params"].get("arguments", {})
            result = handle_mcp_call(tool_name, args)
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    serve()
