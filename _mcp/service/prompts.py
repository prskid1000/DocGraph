"""Prompt definitions and registry for the DocGraph MCP server.

This module provides a centralized registry for all prompts supported in the MCP server.
Each prompt is defined with its configuration including:
- Name and description
- Arguments (if any)
- Message content/template

To add a new prompt:
1. Add a PromptDefinition entry to PROMPT_DEFINITIONS below
2. The handler will automatically use these definitions
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass(frozen=True)
class PromptArgument:
    """Definition of a prompt argument."""
    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class PromptDefinition:
    """Definition of a prompt with all its configuration."""
    
    name: str
    description: str
    arguments: List[PromptArgument]
    message_template: str
    
    def get_message(self, arguments: Optional[Dict] = None) -> str:
        """Get the prompt message, optionally filling in arguments."""
        # For now, just return the template (can be extended for variable substitution)
        return self.message_template


# Prompt definitions registry
PROMPT_DEFINITIONS: Dict[str, PromptDefinition] = {
    "explain_code": PromptDefinition(
        name="explain_code",
        description="Explain what a piece of code does",
        arguments=[
            PromptArgument("file_path", "Path to the file", required=True),
            PromptArgument("line_start", "Starting line number", required=True),
            PromptArgument("line_end", "Ending line number", required=False),
        ],
        message_template="""Please explain what the code at {file_path} (lines {line_start}-{line_end}) does.
        
        Use the DocGraph tools to:
        1. Get the code context
        2. Find related entities and references
        3. Provide a clear, concise explanation"""
    ),
    "find_issues": PromptDefinition(
        name="find_issues",
        description="Analyze code for potential issues and suggest improvements",
        arguments=[
            PromptArgument("file_path", "Path to the file", required=True),
            PromptArgument("entity_name", "Specific function or class to analyze", required=False),
        ],
        message_template="""Please analyze {file_path}{entity_context} for potential issues.
        
        Consider:
        1. Code quality and maintainability
        2. Performance issues
        3. Security concerns
        4. Error handling
        5. Best practices
        
        Use DocGraph tools to understand the code context and dependencies."""
    ),
    "document_entity": PromptDefinition(
        name="document_entity",
        description="Generate documentation for a code entity",
        arguments=[
            PromptArgument("entity_name", "Name of the entity", required=True),
            PromptArgument("entity_type", "Type of entity (function, class, variable)", required=False),
        ],
        message_template="""Please generate comprehensive documentation for {entity_name}{type_context}.
        
        Include:
        1. Description of what it does
        2. Parameters/arguments
        3. Return values
        4. Example usage
        5. Related entities
        6. Notes and edge cases
        
        Use DocGraph tools to gather all necessary information."""
    ),
}


def get_prompt_definition(prompt_name: str) -> Optional[PromptDefinition]:
    """Get a prompt definition by name."""
    return PROMPT_DEFINITIONS.get(prompt_name)


def list_prompts() -> List[PromptDefinition]:
    """Get all prompt definitions."""
    return list(PROMPT_DEFINITIONS.values())


def get_prompt_message(prompt_name: str, arguments: Optional[Dict] = None) -> Optional[str]:
    """Get the message for a prompt with optional argument substitution."""
    prompt = get_prompt_definition(prompt_name)
    if not prompt:
        return None
    
    message = prompt.message_template
    if arguments:
        try:
            message = message.format(**arguments)
        except KeyError:
            # If not all variables are provided, return template as-is
            pass
    
    return message
