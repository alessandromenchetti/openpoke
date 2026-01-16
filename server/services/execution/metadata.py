import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from ...logging_config import logger


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_EXECUTION_META_DIR = _DATA_DIR / "execution_agents"


def _slugify(name: str) -> str:
    """Convert agent name to filesystem-safe slug."""
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "agent"


async def generate_agent_description(agent_name: str) -> Optional[str]:
    """Generate a concise description of an agent based on its request history."""
    from . import get_execution_agent_logs

    logs = get_execution_agent_logs()
    transcript = logs.load_transcript(agent_name)

    if not transcript:
        logger.debug(f"[{agent_name}] No transcript found for description generation.")
        return None

    lines = transcript.split("\n")
    request_lines = [line for line in lines if "<agent_request" in line]

    if not request_lines:
        logger.debug(f"[{agent_name}] No agent requests found in transcript.")
        return None

    request_context = "\n".join(request_lines)

    prompt = f"""Based on this agent's request history, generate a concise description that captures all entities (people, email addresses, topics) it has worked with.

    Agent Name: {agent_name}
    
    Request History:
    {request_context}
    
    Write 1-2 sentences in this format:
    - List all people/email addresses mentioned
    - Include key topics or task types (emails, reminders, scheduled tasks)
    - Be specific and factual
    
    Examples:
    - "Handles emails with Sarah Mitchell (sarah.m@example.com), Alex Chen, and Delta Airlines customer service. Topics include lunch coordination, flight confirmations, and baggage tracking."
    - "Manages communication with Marcus Rivera (m.rivera@techcorp.com) and Elena Vasquez (elena.v@startup.io). Previously sent test messages, searched for project updates, and created reminder triggers."
    - "Creates and manages reminder triggers for code execution checks. Set up scheduled notifications for 2-minute and 5-minute intervals."
    
    Description:"""

    try:
        from ...config import get_settings
        from ...openrouter_client import request_chat_completion

        settings = get_settings()

        response = await request_chat_completion(
            model=settings.summarizer_model,
            messages=[{"role": "user", "content": prompt}],
            system="You're an expert at summarizing agent activities into concise descriptions.",
            api_key=settings.openrouter_api_key,
        )

        description = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if description and len(description) > 20:
            logger.debug(f"[{agent_name}] Generated description: {description}")
            return description

        logger.warning(f"[{agent_name}] Description generation returned empty or too short result {agent_name}.")
        return None

    except Exception as e:
        logger.error(f"[{agent_name}] Failed to generate description: {e}")
        return None


class ExecutionAgentMetadataStore:
    """
    Manages per-agent metadata in .json format.
    """

    def __init__(self, meta_dir: Path):
        self._meta_dir = meta_dir
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._ensure_directory()

    def meta_dir(self) -> Path:
        """Get the metadata directory path."""
        return self._meta_dir

    def _ensure_directory(self) -> None:
        try:
            self._meta_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create directory: {e}")

    def _lock_for(self, agent_name: str) -> threading.Lock:
        """
        Get or create a lock for an agent.
        """
        slug = _slugify(agent_name)
        with self._global_lock:
            if slug not in self._locks:
                self._locks[slug] = threading.Lock()
            return self._locks[slug]

    def _meta_path(self, agent_name: str) -> Path:
        """
        Get meta file path for an agent.
        """
        return self._meta_dir / f"{_slugify(agent_name)}-meta.json"

    def load_metadata(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for an agent.
        """
        path = self._meta_path(agent_name)
        if not path.exists():
            return None
        with self._lock_for(agent_name):
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata for {agent_name}: {e}")
                return None

    def save_metadata(self, agent_name: str, description: Optional[str] = None) -> None:
        """
        Save metadata for an agent.
        """
        path = self._meta_path(agent_name)

        with self._lock_for(agent_name):
            try:
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {
                        "name": agent_name,
                        "description": agent_name,
                        "last_used": None,
                        "usage_count": 0,
                    }

                meta["name"] = agent_name
                if description:
                    meta["description"] = description
                meta["last_used"] = datetime.now(timezone.utc).isoformat()
                meta["usage_count"] = meta.get("usage_count", 0) + 1

                with path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Failed to save metadata for {agent_name}: {e}")

    def clear_all(self) -> None:
        """
        Clear all execution agent metadata files.
        """
        try:
            for meta_file in self._meta_dir.glob("*-meta.json"):
                meta_file.unlink()
            logger.info("Cleared all execution agent metadata")
        except Exception as e:
            logger.error(f"Failed to clear execution metadata: {e}")


_execution_agent_metadata = ExecutionAgentMetadataStore(_EXECUTION_META_DIR)


def get_execution_agent_metadata() -> ExecutionAgentMetadataStore:
    """Get the singleton metadata store instance."""
    return _execution_agent_metadata