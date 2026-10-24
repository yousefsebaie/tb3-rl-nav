import asyncio
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
    print("=== Starting Multi-TurtleBot Plan Generation ===")

    result = asyncio.run(run_python_script("python3 generate_multi_turtlebot_plan.py"))

    if result != 0:
        print("=== Multi-TurtleBot Plan Generation Failed ===")
        return

    print("=== All Planning Stages Completed ===")


if __name__ == "__main__":
    main()