import json
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Any, Tuple, Optional
import os
from pathlib import Path

from scr.utils.extract_agent_view import extract_agent_view
from scr.utils.logger import get_logger
from scr.api.llm_api.providers.completion_result import CompletionResult
import re

logger = get_logger(__name__)

class MessageEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle CompletionResult objects"""
    def default(self, obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)

class Messages(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, messages: List[Dict[str, Any]] = None, filename: str = None, **data):
        # Call Pydantic's initialization
        super().__init__(**data)
        
        # Initialize messages if provided
        if messages:
            self.messages = messages
        
        # Load messages from a file if filename is provided
        if filename:
            self.load(filename)

    def get_last_message(self):
        return self.messages[-1]

    def get_last_message_content(self):
        return self.messages[-1]["content"]

    def get_message(self, index: int):
        return self.messages[index]

    def get_message_content(self, index: int):
        return self.messages[index]["content"]

    def get_message_role(self, index: int):
        return self.messages[index]["role"]

    def append(self, role: str, content: Union[str, CompletionResult]):
        """
        Append a message with the given role and content.
        Content can be a string or a CompletionResult object.
        """
        self.messages.append({"role": role, "content": content})

    def save_to_json(self, filename: str):
        """Save messages in JSON format"""
        # Write to file
        with open(f"{filename}.json", "w") as f:
            json.dump(self.messages, f, indent=4, ensure_ascii=False, cls=MessageEncoder)

    def save_to_markdown(self, filename: str, append: bool = True):
        """
        Save messages in a human-friendly Markdown format.
        Each message is saved with a heading and appropriate formatting.
        JSON content is wrapped in code blocks.
        
        Args:
            filename (str): The filename to save to (without extension)
            append (bool): If True (default), append to existing file instead of overwriting
        """
        def format_content(content):
            if isinstance(content, CompletionResult):
                # Try to format content.content as JSON if possible
                try:
                    content_str = content.content
                    parsed_content = json.loads(content_str)
                    formatted = "#### Content:\n```json\n" + json.dumps(parsed_content, indent=2, ensure_ascii=False) + "\n```\n"
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, use as plain text
                    formatted = "#### Content:\n" + content.content + "\n"
                
                # Add reasoning if present
                if content.reasoning:
                    formatted += "\n#### Reasoning:\n" + content.reasoning
                        
                return formatted
            elif isinstance(content, str):
                try:
                    # Try to parse as JSON for pretty formatting
                    parsed = json.loads(content)
                    return f"```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"
                except json.JSONDecodeError:
                    return content
            else:
                # Content is already a dict or other object
                return content

        # Read existing content if we're appending and the file exists
        existing_lines = []
        system_prompt_already_exists = False
        
        if append and os.path.exists(f"{filename}.md"):
            try:
                with open(f"{filename}.md", "r", encoding='utf-8') as f:
                    existing_lines = f.readlines()
                
                # Check if system prompt already exists in the file
                system_prompt_already_exists = any(line.strip() == "# System Prompt" for line in existing_lines)
                
                # If file exists but is empty or doesn't have the title, add the title
                if not existing_lines or not existing_lines[0].strip().startswith("# Conversation Log"):
                    existing_lines = ["# Conversation Log\n", "\n"] + existing_lines
            except Exception as e:
                logger.warning(f"Failed to read existing markdown file for appending: {e}")
                # Fall back to creating a new file
                existing_lines = ["# Conversation Log\n", "\n"]
        else:
            # New file
            existing_lines = ["# Conversation Log\n", "\n"]
        
        # Create new markdown content for the messages
        markdown_lines = []
        
        # For appending, handle specific multi-step conversation structures if present
        if append and len(self.messages) >= 5:
            # This block logs a specific 5-message sequence:
            # 0: System Prompt
            # 1: Input (Observation) - also source of time step
            # 2: First Reply
            # 3: Reflection Prompt
            # 4: Reflected Refined Reply

            # Only include system prompt if it doesn't already exist in the file
            if not system_prompt_already_exists:
                # Message 0: System Prompt
                msg0_content_raw = self.messages[0].get("content", "")
                system_prompt_display = extract_agent_view(msg0_content_raw)
                if not system_prompt_display or system_prompt_display.isspace():
                    system_prompt_display = format_content(msg0_content_raw)
                system_prompt_display = system_prompt_display.strip()

                markdown_lines.append(f"# System Prompt")
                if system_prompt_display:
                    markdown_lines.append(system_prompt_display)
                markdown_lines.append("") # Blank line for separation

            # Time Step from Message 1 (Input/Observation)
            msg1_content_raw = self.messages[1].get("content", "") # Fetch once for input and time_step
            
            content_for_time_step_regex = ""
            if isinstance(msg1_content_raw, CompletionResult):
                content_for_time_step_regex = msg1_content_raw.content
            elif isinstance(msg1_content_raw, str):
                content_for_time_step_regex = msg1_content_raw
            else: # Fallback if it's some other type, convert to string
                content_for_time_step_regex = str(msg1_content_raw)

            match = re.search(r'Current time step":\s*(\d+)', content_for_time_step_regex)
            time_step = match.group(1) if match else "unknown"

            markdown_lines.append(f"# Current Time Step : {time_step}")
            markdown_lines.append("") # Blank line for separation

            # Message 1: Input
            input_display = extract_agent_view(msg1_content_raw)
            if not input_display or input_display.isspace():
                input_display = format_content(msg1_content_raw)
            input_display = input_display.strip()
            
            markdown_lines.append(f"## Input")
            if input_display:
                markdown_lines.append(input_display)
            markdown_lines.append("") # Blank line for separation

            # Message 2: First Reply
            msg2_content_raw = self.messages[2].get("content", "")
            first_reply_display = format_content(msg2_content_raw).strip()
            markdown_lines.append(f"## First Reply")
            if first_reply_display:
                markdown_lines.append(first_reply_display)
            markdown_lines.append("") # Blank line for separation

            # Message 3: Reflection Prompt
            msg3_content_raw = self.messages[3].get("content", "")
            reflection_prompt_display = extract_agent_view(msg3_content_raw)
            if not reflection_prompt_display or reflection_prompt_display.isspace():
                reflection_prompt_display = format_content(msg3_content_raw)
            reflection_prompt_display = reflection_prompt_display.strip()

            markdown_lines.append(f"## Reflection Prompt")
            if reflection_prompt_display:
                markdown_lines.append(reflection_prompt_display)
            markdown_lines.append("") # Blank line for separation

            # Message 4: Reflected Refined Reply
            msg4_content_raw = self.messages[4].get("content", "")
            refined_reply_display = format_content(msg4_content_raw).strip()
            markdown_lines.append(f"## Reflected Refined Reply")
            if refined_reply_display:
                markdown_lines.append(refined_reply_display)
            markdown_lines.append("") # Blank line for separation
        else:
            # For non-append mode or small conversations, include all messages
            for i, message in enumerate(self.messages, 1):
                role = message.get("role", "unknown").title()
                content = message.get("content", "")
                
                # Add message number and role as heading
                markdown_lines.append(f"## Message {i} - {role}\n\n")
                
                # Format and add content
                formatted_content = format_content(content)
                markdown_lines.append(f"{formatted_content}\n\n")

        # Write to file (append or create new)
        mode = "a" if append else "w"
        with open(f"{filename}.md", mode, encoding='utf-8') as f:
            if append:
                f.write("\n".join(markdown_lines))
            else:
                f.write("\n".join(existing_lines + markdown_lines))

    def save(self, filename: str, output_dir: str = None, append: bool = True):
        """
        Save messages in both JSON and Markdown formats.
        
        Args:
            filename (str): Base filename without extension
            output_dir (str, optional): Directory to save files. If None, uses current directory.
            append (bool): If True (default), append to existing markdown file for agent-based history
        """
        # Create output directory if specified and doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, filename)
        
        # Save both formats
        self.save_to_json(filename)
        self.save_to_markdown(filename, append=append)
        
        logger.info(f"Messages saved to {filename}.json and {filename}.md{' (appended)' if append else ''}")
        
    def save_with_checkpoint_structure(
        self,
        run_id: str,
        time_step: int,
        agent_id: str,
        output_dir: str = "./data"
    ) -> Tuple[str, str]:
        """
        Save messages in the same file structure as checkpoints.
        Saves in both agent-based (appending to history) and step-based directory structures.

        Args:
            run_id (str): The run ID
            time_step (int): The current time step
            agent_id (str): ID of the agent for these messages
            output_dir (str): Base data directory (messages stored under <output_dir>/<run_id>/messages/)

        Returns:
            Tuple[str, str]: Paths to the saved message files (agent-based, step-based)

        Raises:
            OSError: If there's an error creating directories or writing files
        """
        # Convert to Path objects for better path handling
        # Structure: data/<run_id>/messages/
        base_path = Path(output_dir)
        run_path = base_path / run_id / "messages"
        
        # Create base run directory
        run_path.mkdir(parents=True, exist_ok=True)
        
        # Define paths for both structures
        agent_path = run_path / "agent-base" / agent_id
        step_path = run_path / "step-base" / f"step_{time_step}"
        
        # Create directories
        agent_path.mkdir(parents=True, exist_ok=True)
        step_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        # For agent-based, use a single file for all time steps (agent_messages.md)
        agent_base_filename = "agent_messages"
        # For step-based, keep the original format
        step_base_filename = f"messages_t{time_step}_{agent_id}"
        
        agent_output_path = str(agent_path / agent_base_filename)
        step_output_path = str(step_path / step_base_filename)
        
        # Save to both locations with error handling
        try:
            # Save to agent-based location (append mode is now default)
            self.save(agent_output_path)
            
            # Save to step-based location (explicitly set append=False for individual step files)
            self.save(step_output_path, append=False)
            
            logger.info(f"Successfully saved messages to {agent_output_path} (appended) and {step_output_path}")
        except OSError as e:
            logger.error(f"Failed to save messages: {str(e)}")
            raise
            
        return agent_output_path, step_output_path

    def load(self, filename: str):
        """Load messages from a JSON file with enhanced error handling and logging."""
        if not filename:
            raise ValueError("Filename is required")

        file_path = f"{filename}.json"
        logger.info(f"Attempting to load messages from {file_path}")

        try:
            with open(file_path, "r") as f:
                loaded_messages = json.load(f)
        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
            raise ValueError(f"File {file_path} not found")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {file_path}")
            raise ValueError(f"Failed to load messages from {file_path} due to JSON decode error")

        try:
            if not isinstance(loaded_messages, list):
                raise ValueError("Loaded data is not a list of messages")
            # Attempt to assign the loaded messages to self.messages
            self.messages = loaded_messages
        except Exception as e:
            logger.error(f"Failed to assign loaded messages: {e}")
            raise ValueError(f"Error processing loaded messages: {e}")

        logger.info(f"Successfully loaded messages from {file_path}")