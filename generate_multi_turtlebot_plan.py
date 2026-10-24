import json
import os
import re
import cohere

try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}

# Never hard-code the key: put it in config.json (not in git) or COHERE_API_KEY.
COHERE_API_KEY = config.get("cohere_api_key") or os.environ.get("COHERE_API_KEY", "")

client = cohere.Client(COHERE_API_KEY)

mission = """
A collaborative mission is being executed using two TurtleBot3 Burger robots.

TurtleBot3_1 navigates to a supply box located at coordinates (-0.81, 2.5), retrieves the box, and transports it to a handoff location at coordinates (2.0, 2.0).

After TurtleBot3_1 drops the supply box at the handoff location, TurtleBot3_2 navigates to the handoff location, retrieves the supply box, and transports it back to the base.

Both TurtleBot3 robots use LiDAR for navigation and camera detection for locating the supply box.

The mission must generate one shared plan file containing phases for both TurtleBot3_1 and TurtleBot3_2.
"""

example_format = """
{
    "target": "MULTI_TURTLEBOT",
    "phases": [
        {
            "target": "TURTLEBOT3_1",
            "phase_number": 1,
            "state": "TurtleBot3_1 is at base and waiting for the mission start signal.",
            "phase_target": "Wait for mission start.",
            "inputs": [],
            "outputs": [],
            "low_level_plan": "TurtleBot3_1.wait_for_signal()"
        }
    ]
}
"""

prompt = f"""
Generate ONLY valid JSON.

The JSON must EXACTLY follow this structure:

{example_format}

Rules:
- Output ONLY JSON
- No markdown
- No explanations
- Use TurtleBot3_1 and TurtleBot3_2
- Use low_level_plan exactly like the example
- Include:
  - wait_for_signal
  - scan_with_lidar
  - move_to_point
  - detect_with_camera
  - retrieve_object
  - drop_object
  - communicate_with_apm
  - return_to_base
- Generate a complete multi-robot coordinated mission
- Output MUST be valid JSON

Mission:
{mission}
"""

print("Generating AI multi-turtlebot mission...")

response = client.chat(
    model="command-r-plus-08-2024",
    message=prompt,
    temperature=0.1
)

generated = response.text.strip()

generated = re.sub(r"```json", "", generated)
generated = re.sub(r"```", "", generated)

parsed_json = json.loads(generated)

output_path = "/home/g8/Desktop/cyberspace-main/Plans/final_multi_turtlebot_plan.json"

with open(output_path, "w") as f:
    json.dump(parsed_json, f, indent=4)

print("AI-generated multi-turtlebot mission saved successfully")
print("Saved to:", output_path)