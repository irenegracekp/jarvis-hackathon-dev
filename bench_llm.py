"""Quick LLM benchmark: CPU vs GPU on the Gemma model."""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_cpp import Llama

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "google_gemma-3-4b-it-Q4_K_M.gguf")
TEST_PROMPT = [
    {"role": "system", "content": "You are a small witty robot. Respond in JSON with a 'speech' field. Keep it short."},
    {"role": "user", "content": "[What you see: One person waving] [Human says: Hey robot, what do you think about this hackathon?]"},
]

def bench(n_gpu_layers, label):
    print(f"\n{'='*50}")
    print(f"  Benchmark: {label} (n_gpu_layers={n_gpu_layers})")
    print(f"{'='*50}")

    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=n_gpu_layers,
        n_ctx=1024,
        n_batch=128,
        verbose=False,
    )
    t_load = time.time() - t0
    print(f"  Model load: {t_load:.2f}s")

    # Warm-up run
    print("  Warm-up...")
    llm.create_chat_completion(messages=TEST_PROMPT, max_tokens=50, temperature=0.8)

    # Timed runs
    times = []
    for i in range(3):
        t0 = time.time()
        resp = llm.create_chat_completion(
            messages=TEST_PROMPT,
            max_tokens=100,
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        text = resp["choices"][0]["message"]["content"][:80]
        print(f"  Run {i+1}: {elapsed:.2f}s  ->  {text}")

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg:.2f}s")
    return avg

if __name__ == "__main__":
    print(f"Model: {MODEL_PATH}")
    print(f"Exists: {os.path.exists(MODEL_PATH)}")

    cpu_avg = bench(0, "CPU (old default)")
    gpu_avg = bench(-1, "GPU (new default)")

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  CPU average: {cpu_avg:.2f}s")
    print(f"  GPU average: {gpu_avg:.2f}s")
    print(f"  Speedup:     {cpu_avg/gpu_avg:.1f}x")
