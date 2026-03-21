"""Persistent memory store for face-linked person data."""

import json
import os
import threading
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEMORY_PATH = os.path.join(_PROJECT_ROOT, "data", "memories.json")


class MemoryStore:
    def __init__(self, path=DEFAULT_MEMORY_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._data = {}  # face_id -> person dict
        self.load()

    def load(self):
        with self._lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, "r") as f:
                        self._data = json.load(f)
                    print(f"[memory] Loaded {len(self._data)} people from {self.path}")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[memory] Failed to load {self.path}: {e}")
                    self._data = {}
            else:
                self._data = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_person(self, face_id):
        with self._lock:
            return self._data.get(face_id)

    def create_person(self, face_id, name, facts=None, pre_loaded=False):
        with self._lock:
            self._data[face_id] = {
                "name": name,
                "facts": facts or [],
                "last_seen": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "times_seen": 1,
                "pre_loaded": pre_loaded,
            }
            self.save()
        print(f"[memory] Created person: {name} ({face_id})")

    def add_fact(self, face_id, fact):
        with self._lock:
            person = self._data.get(face_id)
            if person is None:
                return
            fact = fact.strip()
            if fact and fact not in person["facts"]:
                person["facts"].append(fact)
                self.save()
                print(f"[memory] Added fact for {person['name']}: {fact}")

    def record_seen(self, face_id):
        with self._lock:
            person = self._data.get(face_id)
            if person is None:
                return
            person["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            person["times_seen"] = person.get("times_seen", 0) + 1
            self.save()

    def set_name(self, face_id, name):
        with self._lock:
            person = self._data.get(face_id)
            if person is None:
                return
            person["name"] = name
            self.save()
            print(f"[memory] Renamed {face_id} -> {name}")

    def get_context_string(self, face_id):
        with self._lock:
            person = self._data.get(face_id)
            if person is None:
                return ""

            name = person["name"]
            facts = person.get("facts", [])
            times = person.get("times_seen", 1)
            last_seen = person.get("last_seen", "")

            parts = [f"You know this person: {name}."]
            if facts:
                parts.append(f"Facts: {', '.join(facts)}.")
            if times > 1:
                parts.append(f"Seen {times} times before.")
            if last_seen:
                parts.append(f"Last seen: {last_seen}.")

            return "[" + " ".join(parts) + "]"

    def list_people(self):
        with self._lock:
            return dict(self._data)


if __name__ == "__main__":
    store = MemoryStore("/tmp/test_memories.json")
    store.create_person("test_1", "Alice", ["Hackathon judge", "Works at NVIDIA"])
    store.record_seen("test_1")
    store.add_fact("test_1", "Likes coffee")
    print(store.get_context_string("test_1"))
    print(store.get_person("test_1"))
    # Cleanup
    os.remove("/tmp/test_memories.json")
    print("Memory test passed!")
