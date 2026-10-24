import requests
import sys

BASE_URL = "http://localhost:5001"

DUMMY_VALUES = {
    "float":  0.5,
    "int":    1,
    "string": "completed",
    "bool":   True,
}

def run_all_phases(robot: str):
    robot = robot.upper()
    print(f"\n=== Running all phases for {robot} ===\n")

    phase = 1
    while True:
        # Get instruction for this phase
        resp = requests.get(f"{BASE_URL}/get_instruction", params={"robot": robot, "phase": phase})

        if resp.status_code == 404:
            print(f"No more phases. {robot} mission complete.")
            break
        elif resp.status_code != 200:
            print(f"Error getting phase {phase}: {resp.status_code} — {resp.text}")
            break

        data = resp.json()
        print(f"--- Phase {phase} ---")
        print(f"Plan:\n{data['low_level_plan']}\n")

        # Build outputs using expected types
        expected = data.get("expected_outputs", {})
        outputs = {
            key: DUMMY_VALUES.get(meta.get("type", "string"), "completed")
            for key, meta in expected.items()
        }
        print(f"Completing with outputs: {outputs}")

        # Complete the phase
        resp = requests.post(f"{BASE_URL}/complete_phase", json={
            "robot": robot,
            "phase": phase,
            "outputs": outputs,
        })

        if resp.status_code != 200:
            print(f"Error completing phase {phase}: {resp.status_code} — {resp.text}")
            break

        print(f"Phase {phase} done: {resp.json()['message']}\n")
        phase += 1

    print("=== Done ===")


if __name__ == "__main__":
    robot = sys.argv[1] if len(sys.argv) > 1 else "ROBOT_DOG"
    run_all_phases(robot)
