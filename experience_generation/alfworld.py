"""
Generate Claude-style experiences for ALFWorld agent using o3 API.

Claude-style experiences have these characteristics:
1. Concise and actionable - focus on what to do, not verbose descriptions
2. General principles - transferable across similar tasks
3. Failure-aware - include common pitfalls and how to avoid them
4. Hierarchical - general experiences + task-specific experiences

Output format:
{
    "general_experiences": [
        {
            "experience_id": "...",
            "title": "...",  # Short title
            "principle": "...",  # The core insight/rule
            "when_to_apply": "...",  # Triggering condition
            "example": "..."  # Brief concrete example (optional)
        }
    ],
    "task_specific_experiences": {
        "pick_and_place": [...],
        "look_at_obj_in_light": [...],
        "clean": [...],
        "heat": [...],
        "cool": [...],
        "examine": [...]
    },
    "common_mistakes": [
        {
            "mistake_id": "...",
            "description": "...",  # What the mistake is
            "why_it_happens": "...",  # Why agents make this mistake
            "how_to_avoid": "..."  # Concrete fix
        }
    ]
}
"""

import json
import os
from typing import List, Dict, Any
from openai import AzureOpenAI

class OpenAIClient:
    def __init__(self, max_new_tokens: int = 4096, model: str = "o3"):
        self.max_new_tokens = max_new_tokens
        self.model = model
        self.client = AzureOpenAI(
            api_key="",
            azure_endpoint="",
            api_version=""
        )

    def generate_response(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content


def load_memories(json_path: str) -> List[Dict]:
    """Load existing memory data."""
    with open(json_path, 'r') as f:
        return json.load(f)


def categorize_by_task_type(memories: List[Dict]) -> Dict[str, Dict[str, List]]:
    """Categorize memories by task type and outcome."""
    categorized = {
        'pick_and_place': {'success': [], 'failure': []},
        'look_at_obj_in_light': {'success': [], 'failure': []},
        'clean': {'success': [], 'failure': []},
        'heat': {'success': [], 'failure': []},
        'cool': {'success': [], 'failure': []},
        'examine': {'success': [], 'failure': []},
    }

    for mem in memories:
        goal = mem['content']['task_meta']['original_goal'].lower()
        outcome = 'success' if mem['tags']['outcome'] == 'Success' else 'failure'

        # Categorize based on goal text
        if 'look at' in goal and 'under' in goal:
            task_type = 'look_at_obj_in_light'
        elif 'clean' in goal:
            task_type = 'clean'
        elif 'heat' in goal:
            task_type = 'heat'
        elif 'cool' in goal:
            task_type = 'cool'
        elif 'examine' in goal or 'find' in goal:
            task_type = 'examine'
        elif 'put' in goal:
            task_type = 'pick_and_place'
        else:
            continue  # Skip unknown types

        categorized[task_type][outcome].append(mem)

    return categorized


def extract_patterns(memories: List[Dict]) -> str:
    """Extract key patterns from memories for prompt."""
    patterns = []
    for mem in memories[:10]:  # Limit to 10 for context
        goal = mem['content']['task_meta']['original_goal']
        refined_traj = mem['content'].get('refined_trajectory') or {}
        trajectory = refined_traj.get('refined_trajectory', []) if refined_traj else []
        strategic = mem['content'].get('strategic_guidelines') or {}
        mistakes = strategic.get('mistakes_to_avoid', []) if strategic else []
        planning = strategic.get('planning_pattern', '') if strategic else ''

        pattern = {
            'goal': goal,
            'steps': [{'action': s.get('action', ''), 'reasoning': s.get('reasoning', '')}
                     for s in trajectory[:5]],
            'planning_pattern': planning,
            'mistakes': mistakes[:3] if mistakes else []
        }
        patterns.append(pattern)

    return json.dumps(patterns, indent=2)


def generate_general_experiences(client: OpenAIClient, categorized_memories: Dict) -> List[Dict]:
    """Generate general experiences using o3."""

    # Collect sample data across all task types
    all_successes = []
    all_failures = []
    for task_type, data in categorized_memories.items():
        all_successes.extend(data['success'][:5])
        all_failures.extend(data['failure'][:5])

    success_patterns = extract_patterns(all_successes)
    failure_patterns = extract_patterns(all_failures)

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable experiences.

Analyze these successful and failed trajectories from an embodied AI agent operating in household environments (ALFWorld).

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 8-12 GENERAL EXPERIENCES that apply across ALL task types. These should be:
1. **Concise** - Each experience should be 1-2 sentences max
2. **Actionable** - Clear what to do, not vague principles
3. **Transferable** - Apply to pick_and_place, heat, cool, clean, examine, look_at_obj_in_light tasks
4. **Failure-aware** - Derived from what went wrong in failures

Format as JSON array:
[
    {{
        "experience_id": "gen_001",
        "title": "Short title (3-5 words)",
        "principle": "The core actionable insight in 1-2 sentences",
        "when_to_apply": "Specific trigger condition"
    }}
]

Focus on:
- Navigation and exploration strategies
- Object manipulation principles
- State tracking and goal decomposition
- Error recovery patterns
- Container/furniture interaction rules

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    # Parse JSON from response
    try:
        # Try to extract JSON from response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def generate_task_specific_experiences(client: OpenAIClient, task_type: str,
                                   successes: List[Dict], failures: List[Dict]) -> List[Dict]:
    """Generate task-specific experiences for a particular task type."""

    if not successes and not failures:
        return []

    success_patterns = extract_patterns(successes[:8])
    failure_patterns = extract_patterns(failures[:8]) if failures else "[]"

    task_descriptions = {
        'pick_and_place': 'Pick up object(s) from one location and place them at a target location',
        'look_at_obj_in_light': 'Find an object and examine it under a light source (usually desklamp)',
        'clean': 'Find an object, clean it in a sink/basin, then place it somewhere',
        'heat': 'Find an object, heat it in microwave, then place it somewhere',
        'cool': 'Find an object, cool it in fridge, then place it somewhere',
        'examine': 'Find and examine a specific object'
    }

    prompt = f"""You are an expert at distilling agent behavior patterns into concise, actionable experiences.

Task Type: {task_type.upper()}
Description: {task_descriptions.get(task_type, '')}

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 4-6 TASK-SPECIFIC EXPERIENCES for {task_type} tasks. These should be:
1. **Concise** - 1-2 sentences max per experience
2. **Specific** - Apply specifically to {task_type} tasks
3. **Actionable** - Clear steps or decision rules
4. **Pattern-based** - Identify what makes success vs failure

Format as JSON array:
[
    {{
        "experience_id": "{task_type[:3]}_001",
        "title": "Short title (3-5 words)",
        "principle": "The core actionable insight",
        "when_to_apply": "Specific trigger condition"
    }}
]

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def generate_common_mistakes(client: OpenAIClient, categorized_memories: Dict) -> List[Dict]:
    """Generate common mistakes to avoid."""

    # Collect all failure patterns
    all_failures = []
    for task_type, data in categorized_memories.items():
        for mem in data['failure'][:5]:
            mistakes = mem['content'].get('strategic_guidelines', {}).get('mistakes_to_avoid', [])
            if mistakes:
                all_failures.append({
                    'task_type': task_type,
                    'goal': mem['content']['task_meta']['original_goal'],
                    'mistakes': mistakes[:3]
                })

    failure_data = json.dumps(all_failures[:15], indent=2)

    prompt = f"""You are an expert at analyzing agent failures and distilling them into avoidable mistakes.

Analyze these failure patterns from an embodied AI agent:

{failure_data}

Generate 8-12 COMMON MISTAKES to avoid. Format as JSON array:
[
    {{
        "mistake_id": "err_001",
        "description": "What the mistake is (1 sentence)",
        "why_it_happens": "Why agents make this mistake (1 sentence)",
        "how_to_avoid": "Concrete actionable fix (1-2 sentences)"
    }}
]

Focus on:
- Exploration failures (getting stuck, not finding objects)
- State management errors (forgetting what you're holding)
- Goal misunderstanding (wrong object, incomplete task)
- Inefficient action sequences

Return ONLY the JSON array, no other text."""

    response = client.generate_response([
        {"role": "user", "content": prompt}
    ])

    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return []


def main():
    # Configuration
    memory_json_path = os.path.expanduser(
        "memory_data/alfworld/generated_memories_alfworld_total.json"
    )
    output_path = os.path.expanduser(
        "memory_data/alfworld/claude_style_experiences.json"
    )

    print("Loading existing memories...")
    memories = load_memories(memory_json_path)
    print(f"Loaded {len(memories)} memories")

    print("\nCategorizing by task type...")
    categorized = categorize_by_task_type(memories)
    for task_type, data in categorized.items():
        print(f"  {task_type}: {len(data['success'])} success, {len(data['failure'])} failure")

    # Initialize client
    client = OpenAIClient(max_new_tokens=4096, model="o3")

    # Generate experiences
    print("\n=== Generating General Experiences ===")
    general_experiences = generate_general_experiences(client, categorized)
    print(f"Generated {len(general_experiences)} general experiences")

    print("\n=== Generating Task-Specific Experiences ===")
    task_specific_experiences = {}
    for task_type, data in categorized.items():
        print(f"  Processing {task_type}...")
        experiences = generate_task_specific_experiences(
            client, task_type,
            data['success'], data['failure']
        )
        task_specific_experiences[task_type] = experiences
        print(f"    Generated {len(experiences)} experiences")

    print("\n=== Generating Common Mistakes ===")
    common_mistakes = generate_common_mistakes(client, categorized)
    print(f"Generated {len(common_mistakes)} mistakes")

    # Compile final output
    output = {
        "general_experiences": general_experiences,
        "task_specific_experiences": task_specific_experiences,
        "common_mistakes": common_mistakes,
        "metadata": {
            "source": "generated from ALFWorld trajectories using o3",
            "total_memories_analyzed": len(memories),
            "task_distribution": {
                task_type: {
                    "success": len(data['success']),
                    "failure": len(data['failure'])
                }
                for task_type, data in categorized.items()
            }
        }
    }

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary ===")
    print(f"General experiences: {len(general_experiences)}")
    print(f"Task-specific experiences: {sum(len(s) for s in task_specific_experiences.values())}")
    print(f"Common mistakes: {len(common_mistakes)}")
    print(f"\nSaved to: {output_path}")

    # Print sample output
    print("\n=== Sample General Experiences ===")
    for experience in general_experiences[:3]:
        print(f"\n[{experience.get('experience_id', 'N/A')}] {experience.get('title', 'N/A')}")
        print(f"  {experience.get('principle', 'N/A')}")


if __name__ == "__main__":
    main()
