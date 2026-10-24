import asyncio
import json
import warnings

warnings.filterwarnings("ignore")


async def run_python_script(command):
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if stdout:
        print(stdout.decode())

    if stderr:
        print(stderr.decode())

    return process.returncode


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    robots = config.get("robots_in_curr_mission", [])

    print("=== Step 1: High-Level Plan Generation ===")
    result = asyncio.run(run_python_script("python3 High_Level_Plan_Generator.py"))
    if result != 0:
        print("=== High-Level Plan Generation Failed ===")
        return

    print("=== Step 2: Low-Level Planning per Robot ===")
    for robot in robots:
        print(f"--- Low-Level Planner: {robot} ---")
        result = asyncio.run(run_python_script(f"python3 Low_Level_Planner.py {robot}"))
        if result != 0:
            print(f"=== Low-Level Planning Failed for {robot} ===")
            return

    print("=== All Planning Stages Completed ===")


if __name__ == "__main__":
    main()
