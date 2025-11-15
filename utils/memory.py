import json
from datetime import datetime
import logging


class SessionMemory:
    """
    Manages temporary session state during agent workflow execution.
    Data is lost when session ends.
    """

    def __init__(self):
        self.state = {}
        self.history = []
        self.logger = logging.getLogger(__name__)

    def save(self, key, value):
        """Save key-value pair to session memory."""
        self.state[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.debug(f"Saved to session memory: {key}")

    def load(self, key):
        """Retrieve value from session memory."""
        entry = self.state.get(key)
        if entry:
            return entry['value']
        return None

    def add_to_history(self, event):
        """Log an event to session history."""
        self.history.append({
            'event': event,
            'timestamp': datetime.now().isoformat()
        })

    def get_history(self):
        """Retrieve session history."""
        return self.history

    def clear(self):
        """Clear all session data."""
        self.state = {}
        self.history = []
        self.logger.info("Session memory cleared")


class LongTermMemory:
    """
    Manages persistent storage of analysis results and learned patterns.
    Simulates persistence using JSON files.
    """

    def __init__(self, storage_path='data/long_term_memory.json'):
        self.storage_path = storage_path
        self.memory_store = self._load_memory()
        self.logger = logging.getLogger(__name__)

    def _load_memory(self):
        """Load existing memory from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            self.logger.warning("Corrupted memory file, starting fresh")
            return {}

    def _save_memory(self):
        """Persist memory to disk."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.memory_store, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save memory: {str(e)}")

    def remember(self, key, value):
        """Store information in long-term memory."""
        self.memory_store[key] = {
            'value': value,
            'created_at': datetime.now().isoformat(),
            'access_count': 0
        }
        self._save_memory()
        self.logger.info(f"Stored in long-term memory: {key}")

    def recall(self, key):
        """Retrieve information from long-term memory."""
        entry = self.memory_store.get(key)
        if entry:
            entry['access_count'] += 1
            entry['last_accessed'] = datetime.now().isoformat()
            self._save_memory()
            return entry['value']
        return None

    def forget(self, key):
        """Remove entry from long-term memory."""
        if key in self.memory_store:
            del self.memory_store[key]
            self._save_memory()
            self.logger.info(f"Removed from long-term memory: {key}")

    def get_all_keys(self):
        """List all stored keys."""
        return list(self.memory_store.keys())
