"""
AI Dataset Radar - Agent Integration Module

This module provides tools for AI agents to consume radar intelligence:

- schema.json: JSON Schema for report format validation
- tools.json: Tool definitions for OpenAI/Anthropic function calling
- prompts.md: Pre-built system prompts for agent integration
- api.py: FastAPI REST server for HTTP access

Quick Start:

1. HTTP API (for any agent):
   ```bash
   pip install fastapi uvicorn
   uvicorn agent.api:app --port 8080
   ```

2. Function Calling (OpenAI/Anthropic):
   Load tools from agent/tools.json

3. JSON Consumption:
   Validate reports against agent/schema.json

4. MCP Server (Claude Desktop):
   See mcp_server/server.py
"""

__version__ = "1.0.0"
