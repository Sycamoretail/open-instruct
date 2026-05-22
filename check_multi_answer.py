"""Check multiple <answer> tags in SFT data."""
import json
import re

data = []
with open("data/scholar/sft_distill_v2.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

in_think_count = 0
real_multi = 0
real_multi_examples = []

for item in data:
    msgs = item["messages"]
    # Skip system prompt
    non_sys = [m for m in msgs if m["role"] != "system"]

    # Combine all assistant messages
    assistant_parts = [m["content"] for m in non_sys if m["role"] == "assistant"]
    full_assistant = "\n".join(assistant_parts)

    # Count <answer> in full text
    answer_opens = len(re.findall(r"<answer>", full_assistant))
    if answer_opens <= 1:
        continue

    # Remove all <think>...</think> content, then count <answer>
    no_think = re.sub(r"<think>.*?</think>", "", full_assistant, flags=re.DOTALL)
    answer_outside_think = len(re.findall(r"<answer>", no_think))

    if answer_outside_think > 1:
        real_multi += 1
        if len(real_multi_examples) < 5:
            # Find positions of <answer> outside think
            positions = []
            for match in re.finditer(r"<answer>", no_think):
                start = max(0, match.start() - 50)
                end = min(len(no_think), match.end() + 200)
                positions.append(no_think[start:end])
            real_multi_examples.append({
                "id": item.get("example_id", item.get("id", "?")),
                "answer_total": answer_opens,
                "answer_outside_think": answer_outside_think,
                "positions": positions,
            })
    else:
        in_think_count += 1

print(f"Total samples with multiple <answer> (excl system prompt): {in_think_count + real_multi}")
print(f"  - Extra <answer> ONLY inside <think>: {in_think_count} (harmless)")
print(f"  - Multiple <answer> OUTSIDE <think>: {real_multi} (problematic)")
print()

if real_multi_examples:
    print("=== Examples of problematic cases (multiple <answer> outside <think>) ===")
    for i, ex in enumerate(real_multi_examples):
        print(f"\n--- Example {i+1} ---")
        print(f"ID: {ex['id']}, total <answer>={ex['answer_total']}, outside <think>={ex['answer_outside_think']}")
        for j, pos in enumerate(ex["positions"]):
            print(f"  Position {j+1}: ...{repr(pos[:250])}...")
        print()
else:
    print("No problematic cases found - all extra <answer> tags are inside <think> blocks.")

# Additional check: are there unclosed <think> blocks that might hide <answer>?
print("\n=== Additional: check for unclosed <think> blocks ===")
unclosed_think = 0
for item in data:
    msgs = item["messages"]
    non_sys = [m for m in msgs if m["role"] != "system"]
    assistant_parts = [m["content"] for m in non_sys if m["role"] == "assistant"]
    full_assistant = "\n".join(assistant_parts)
    
    think_opens = len(re.findall(r"<think>", full_assistant))
    think_closes = len(re.findall(r"</think>", full_assistant))
    if think_opens != think_closes:
        unclosed_think += 1

print(f"Samples with unclosed <think> blocks: {unclosed_think}/{len(data)}")

# Check: multi-turn conversations where model gives intermediate answers
print("\n=== Check: multi-answer as multi-turn intermediate answers ===")
multi_turn_answer = 0
for item in data:
    msgs = item["messages"]
    non_sys = [m for m in msgs if m["role"] != "system"]
    # Count how many assistant messages contain <answer>
    asst_with_answer = 0
    for m in non_sys:
        if m["role"] == "assistant" and "<answer>" in m["content"]:
            asst_with_answer += 1
    if asst_with_answer > 1:
        multi_turn_answer += 1

print(f"Samples where multiple assistant messages each have <answer>: {multi_turn_answer}/{len(data)}")
