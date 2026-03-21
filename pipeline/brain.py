"""LLM orchestration -- takes speech + vision context, returns structured JSON

Supports three backends:
  - "local"      : llama-cpp-python with a local GGUF model (default)
  - "openai"     : OpenAI API (gpt-4o-mini) — requires OPENAI_API_KEY env var
  - "openrouter" : OpenRouter API — requires OPENROUTER_API_KEY env var
"""

import json
import os
import random
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")

# Default GGUF model filename — update after download
DEFAULT_MODEL = "google_gemma-3-4b-it-Q4_K_M.gguf"

# OpenAI settings
OPENAI_MODEL = "gpt-4o-mini"

# OpenRouter settings
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _load_system_prompt():
    path = os.path.join(CONFIG_DIR, "system_prompt.txt")
    with open(path, "r") as f:
        return f.read().strip()


def _load_ambient_lines():
    path = os.path.join(CONFIG_DIR, "ambient_lines.json")
    with open(path, "r") as f:
        return json.load(f)


def _default_response(speech, emotion="calm"):
    return {
        "speech": speech,
        "emotion": emotion,
        "head_direction": "scanning",
        "antenna_state": "neutral",
    }


class BrainPipeline:
    def __init__(self, model_name=None, n_gpu_layers=0, n_ctx=1024, memory_store=None,
                 llm_backend="local"):
        self.llm_backend = llm_backend
        self.model_path = os.path.join(MODEL_DIR, model_name or DEFAULT_MODEL)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.memory = memory_store

        self.system_prompt = _load_system_prompt()
        self.ambient_lines = _load_ambient_lines()

        self._llm = None
        self._openai_client = None
        self._conversations = {}  # face_id -> list of messages
        self._last_ambient_idx = -1

    def _load_llm(self):
        if self.llm_backend == "openai":
            return self._load_openai()
        if self.llm_backend == "openrouter":
            return self._load_openrouter()
        return self._load_local_llm()

    def _load_openai(self):
        if self._openai_client is not None:
            return True
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("[brain] OPENAI_API_KEY not set.")
                return False
            self._openai_client = OpenAI(api_key=api_key)
            print(f"[brain] OpenAI client ready (model={OPENAI_MODEL}).")
            return True
        except ImportError:
            print("[brain] openai package not installed. pip install openai")
            return False
        except Exception as e:
            print(f"[brain] Failed to init OpenAI client: {e}")
            return False

    def _load_openrouter(self):
        if self._openai_client is not None:
            return True
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                print("[brain] OPENROUTER_API_KEY not set.")
                return False
            self._openai_client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
            )
            print(f"[brain] OpenRouter client ready (model={OPENROUTER_MODEL}).")
            return True
        except ImportError:
            print("[brain] openai package not installed. pip install openai")
            return False
        except Exception as e:
            print(f"[brain] Failed to init OpenRouter client: {e}")
            return False

    def _load_local_llm(self):
        if self._llm is not None:
            return True
        try:
            from llama_cpp import Llama

            if not os.path.exists(self.model_path):
                print(f"[brain] Model not found: {self.model_path}")
                print("[brain] Available models:", os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else "none")
                return False

            print(f"[brain] Loading LLM from {self.model_path} (gpu_layers={self.n_gpu_layers})...")
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=128,
                verbose=False,
            )
            print("[brain] LLM loaded.")
            return True
        except Exception as e:
            print(f"[brain] Failed to load LLM: {e}")
            return False

    def _chat_completion(self, messages, max_tokens=200, temperature=0.8):
        """Unified chat completion for all backends. Returns raw text."""
        if self.llm_backend in ("openai", "openrouter"):
            model = OPENROUTER_MODEL if self.llm_backend == "openrouter" else OPENAI_MODEL
            response = self._openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        else:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            return response["choices"][0]["message"]["content"]

    def ambient_react(self, scene_description=""):
        """Pick an ambient line. Zero LLM latency."""
        # Cycle through ambient lines, avoiding repeats
        self._last_ambient_idx = (self._last_ambient_idx + 1) % len(self.ambient_lines)
        # Occasionally shuffle for variety
        if self._last_ambient_idx == 0:
            random.shuffle(self.ambient_lines)
        line = self.ambient_lines[self._last_ambient_idx]

        emotions = ["amused", "curious", "calm", "skeptical"]
        return _default_response(line, random.choice(emotions))

    def engage(self, speech_text, scene_description="", face_id="default", memory_context=""):
        """Full LLM call for engaged conversation."""
        if not self._load_llm():
            return _default_response(
                "Hmm, my brain is still warming up. Give me a second.",
                "surprised",
            )

        # Get or create conversation history
        if face_id not in self._conversations:
            self._conversations[face_id] = []
        history = self._conversations[face_id]

        # Build context message
        context_parts = []
        if memory_context:
            context_parts.append(memory_context)
        if scene_description:
            context_parts.append(f"[What you see: {scene_description}]")
        context_parts.append(f"[Human says: {speech_text}]")
        user_msg = "\n".join(context_parts)

        # Build messages for the LLM
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add recent history (last 6 turns to save context)
        for msg in history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_msg})

        try:
            raw = self._chat_completion(messages, max_tokens=200, temperature=0.8)

            # Parse JSON response
            result = self._parse_response(raw)

            # Update conversation history
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": raw})

            # Trim history if too long
            if len(history) > 20:
                self._conversations[face_id] = history[-12:]

            return result

        except Exception as e:
            print(f"[brain] LLM error: {e}")
            return _default_response(
                "Sorry, I got a bit tangled in my own wires there. What were you saying?",
                "surprised",
            )

    def greet(self, face_id, person_info, scene_description=""):
        """Generate a personalized greeting for an arriving face."""
        if person_info:
            name = person_info.get("name", "someone")
            facts = person_info.get("facts", [])
            times = person_info.get("times_seen", 1)
            facts_str = ", ".join(facts) if facts else "no known details"

            if not self._load_llm():
                # Template fallback
                if times > 1:
                    return _default_response(f"Hey {name}! Good to see you again.", "excited")
                else:
                    return _default_response(f"Well hello, {name}! Nice to finally meet in person.", "curious")

            prompt = (
                f"[A person just arrived in front of you. You recognize them as {name}. "
                f"What you know about them: {facts_str}. "
                f"You've seen them {times} time(s) before. "
                f"Scene: {scene_description or 'someone approaching'}]\n"
                f"Greet them naturally — use their name, reference something you know about them if relevant. "
                f"Keep it short and warm."
            )
        else:
            if not self._load_llm():
                return _default_response("Oh, hello there! I don't think we've met. What's your name?", "curious")

            prompt = (
                f"[A new person just appeared in front of you. You don't recognize them. "
                f"Scene: {scene_description or 'someone approaching'}]\n"
                f"Greet them with curiosity. You're intrigued by new faces. Maybe ask their name."
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._chat_completion(messages, max_tokens=150, temperature=0.9)
            result = self._parse_response(raw)
            # Ensure greeting isn't empty
            if not result.get("speech", "").strip():
                print(f"[brain] Greet LLM returned empty speech, raw: {raw[:200]}")
                if person_info:
                    return _default_response(f"Hey {person_info.get('name', 'there')}!", "excited")
                return _default_response("Oh, hello there! I don't think we've met. What's your name?", "curious")
            return result
        except Exception as e:
            print(f"[brain] Greet LLM error: {e}")
            if person_info:
                return _default_response(f"Hey {person_info.get('name', 'there')}!", "excited")
            return _default_response("Oh, hello! New face alert!", "curious")

    def _parse_response(self, raw_text):
        """Parse LLM JSON output, with fallbacks."""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw_text[start:end])
                except json.JSONDecodeError:
                    return _default_response(raw_text[:200])
            else:
                return _default_response(raw_text[:200])

        valid_emotions = {"excited", "curious", "calm", "surprised", "amused", "skeptical"}
        valid_heads = {"toward_speaker", "scanning", "tilt_left", "tilt_right", "nod"}
        valid_antennas = {"perked", "wiggle", "drooped", "neutral"}

        result = {
            "speech": str(data.get("speech", "...")),
            "emotion": data.get("emotion", "calm") if data.get("emotion") in valid_emotions else "calm",
            "head_direction": data.get("head_direction", "toward_speaker") if data.get("head_direction") in valid_heads else "toward_speaker",
            "antenna_state": data.get("antenna_state", "neutral") if data.get("antenna_state") in valid_antennas else "neutral",
        }

        # Extract optional save_memory field
        save_mem = data.get("save_memory")
        if save_mem and isinstance(save_mem, str) and save_mem.strip():
            result["save_memory"] = save_mem.strip()

        return result

    def clear_conversation(self, face_id="default"):
        self._conversations.pop(face_id, None)

    def clear_all_conversations(self):
        self._conversations.clear()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-backend", default="local", choices=["local", "openai"])
    args = parser.parse_args()

    bp = BrainPipeline(llm_backend=args.llm_backend)

    # Test ambient mode (no LLM needed)
    for i in range(3):
        r = bp.ambient_react("Two people standing nearby")
        print(f"Ambient {i}: {r}")

    # Test engaged mode (needs LLM)
    print(f"\nTesting engaged mode (backend={args.llm_backend})...")
    r = bp.engage("Hey robot, what do you think about this hackathon?", "A person waving at the camera")
    print(f"Engaged: {r}")
