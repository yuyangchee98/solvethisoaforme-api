"""MCP tool definitions for the agent system."""

from pathlib import Path

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool

from .patent_fetcher import fetch_patent


def create_patent_tools_server(workspace: Path) -> McpSdkServerConfig:
    """Create an MCP server with patent-related tools.

    The workspace path is captured in the tool closure so the tool
    knows where to save fetched patents.
    """

    @tool(
        "FetchPatent",
        "Fetch a patent or patent application from Google Patents by publication number. "
        "Saves the full text (title, abstract, description, claims) as a markdown file "
        "in input/. Supports formats like 'US 2022/0075747 A1', 'US20220075747A1', "
        "'US 11,234,567 B2'. Returns the file path and metadata on success.",
        {"publication_number": str},
    )
    async def fetch_patent_tool(args: dict) -> dict:
        pub_number = args.get("publication_number", "")
        if not isinstance(pub_number, str) or not pub_number.strip():
            return {
                "content": [{"type": "text", "text": "Error: publication_number is required"}],
                "is_error": True,
            }
        result = await fetch_patent(pub_number.strip(), workspace)

        if result.success:
            text = (
                f"Fetched: {result.title}\n"
                f"Publication: {result.publication_number}\n"
                f"Claims: {result.claim_count}\n"
                f"Saved to: {result.file_path}"
            )
            return {"content": [{"type": "text", "text": text}]}
        else:
            return {
                "content": [{"type": "text", "text": f"Error: {result.error}"}],
                "is_error": True,
            }

    return create_sdk_mcp_server(
        name="patent-tools",
        tools=[fetch_patent_tool],
    )
