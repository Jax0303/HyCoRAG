"""
Script to download and test Qwen2.5-3B model.
"""
import sys
sys.path.insert(0, '/home/user/HyCoRAG')

from hycorag.rag.llm_client import LocalLLMClient

print("Downloading Qwen2.5-3B-Instruct...")
print("This may take several minutes (model size: ~6GB)")

# Initialize client (will download model on first run)
client = LocalLLMClient(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto"
)

# Test generation
test_prompt = """Given the following table context:

Year | Revenue | Profit
2020 | $100M | $20M
2021 | $150M | $30M
2022 | $200M | $40M

Question: What was the revenue in 2021?"""

print("\n" + "="*50)
print("Testing LLM generation...")
print("="*50)

response = client.generate(test_prompt, max_tokens=100)

print(f"\nPrompt:\n{test_prompt}")
print(f"\nResponse:\n{response}")
print("\n" + "="*50)
print("LLM test complete!")
