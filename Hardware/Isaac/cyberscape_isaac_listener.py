"""Isaac Sim Script Editor listener for CyberScape TurtleBot3 plans.

Run inside Isaac Sim:
exec(open("/home/g8/Downloads/cyberspace-main/Hardware/Isaac/cyberscape_isaac_listener.py").read())
"""

from pxr import Gf, UsdGeom
import math
import omni.kit.app
import omni.usd
import os
import re
import time


COMMAND_FILE = os.environ.get("CYBERSCAPE_ISAAC_COMMAND_FILE", "/home/g8/cyberscape_command.txt")
STATUS_FILE = os.environ.get("CYBERSCAPE_ISAAC_STATUS_FILE", "/home/g8/cyberscape_status.txt")
ROBOT_PATH = os.environ.get("CYBERSCAPE_ISAAC_ROBOT_PATH", "/World/turtlebot3_burger_02")

STEP_DELAY = 0.20
STEP_SIZE = 0.05
ROTATION_DELAY = 0.15
ROTATION_STEP_DEGREES = 5.0

last_mtime = os.path.getmtime(COMMAND_FILE) if os.path.exists(COMMAND_FILE) else 0
path_steps = []
rotation_steps = []
active_movement = False
active_rotation = False
last_move_step_time = 0
last_rotation_step_time = 0
base_x = 0.0
base_y = 0.0


def write_status(status):
    with open(STATUS_FILE, "w") as file:
        file.write(status)


def get_stage():
    return omni.usd.get_context().get_stage()


def get_prim(path):
    return get_stage().GetPrimAtPath(path)


def get_translate(path):
    prim = get_prim(path)
    if not prim.IsValid():
        return None

    attr = prim.GetAttribute("xformOp:translate")
    if not attr or not attr.HasValue():
        return None

    pos = attr.Get()
    return float(pos[0]), float(pos[1]), float(pos[2])


def set_translate(path, x, y, z=None):
    prim = get_prim(path)
    if not prim.IsValid():
        print("Isaac CyberScape: prim not found:", path)
        return False

    attr = prim.GetAttribute("xformOp:translate")
    if not attr or not attr.HasValue():
        xform = UsdGeom.Xformable(prim)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(x), float(y), 0.0 if z is None else float(z)))
        return True

    old_pos = attr.Get()
    attr.Set(Gf.Vec3d(float(x), float(y), old_pos[2] if z is None else float(z)))
    return True


def get_robot_position():
    return get_translate(ROBOT_PATH)


def set_robot_position(x, y):
    return set_translate(ROBOT_PATH, x, y)


def get_or_create_rotate_op():
    prim = get_prim(ROBOT_PATH)
    if not prim.IsValid():
        print("Isaac CyberScape: robot path not found:", ROBOT_PATH)
        return None

    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            return op

    return xform.AddRotateXYZOp()


def get_robot_rotation_z():
    rotate_op = get_or_create_rotate_op()
    if rotate_op is None:
        return 0.0

    value = rotate_op.Get()
    if value is None:
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        return 0.0

    return float(value[2])


def set_robot_rotation_z(z_degrees):
    rotate_op = get_or_create_rotate_op()
    if rotate_op is None:
        return

    value = rotate_op.Get() or Gf.Vec3f(0.0, 0.0, 0.0)
    rotate_op.Set(Gf.Vec3f(float(value[0]), float(value[1]), float(z_degrees)))


def parse_command(command):
    match = re.match(r"\s*([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\((.*)\)\s*", command)
    if match is None:
        return "", ""
    return match.group(2), match.group(3)


def parse_move_args(args):
    match = re.search(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", args)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def parse_rotate_args(args):
    match = re.search(r"-?\d+(?:\.\d+)?", args)
    if match is None:
        return None
    return float(match.group(0))


def build_step_path(start_x, start_y, target_x, target_y):
    result = []
    current_x = float(start_x)
    current_y = float(start_y)
    target_x = float(target_x)
    target_y = float(target_y)

    while abs(current_x - target_x) > STEP_SIZE:
        current_x += STEP_SIZE if target_x > current_x else -STEP_SIZE
        result.append((round(current_x, 2), round(current_y, 2)))

    current_x = target_x
    result.append((round(current_x, 2), round(current_y, 2)))

    while abs(current_y - target_y) > STEP_SIZE:
        current_y += STEP_SIZE if target_y > current_y else -STEP_SIZE
        result.append((round(current_x, 2), round(current_y, 2)))

    current_y = target_y
    result.append((round(current_x, 2), round(current_y, 2)))
    return result


def build_rotation_path(start_angle, target_angle):
    result = []
    current_angle = float(start_angle)
    target_angle = float(target_angle)

    while abs(current_angle - target_angle) > ROTATION_STEP_DEGREES:
        current_angle += ROTATION_STEP_DEGREES if target_angle > current_angle else -ROTATION_STEP_DEGREES
        result.append(round(current_angle, 2))

    result.append(round(target_angle, 2))
    return result


def start_move_to_point(target_x, target_y):
    global active_movement, path_steps
    current_position = get_robot_position()
    if current_position is None:
        print("Isaac CyberScape: robot position unavailable")
        write_status("DONE")
        return

    start_x, start_y, _ = current_position
    path_steps = build_step_path(start_x, start_y, target_x, target_y)
    active_movement = True
    print("Isaac CyberScape: moving to", target_x, target_y, "steps", len(path_steps))


def start_rotation(angle_degrees):
    global active_rotation, rotation_steps
    start_angle = get_robot_rotation_z()
    target_angle = start_angle + angle_degrees
    rotation_steps = build_rotation_path(start_angle, target_angle)
    active_rotation = True
    print("Isaac CyberScape: rotating by", angle_degrees, "steps", len(rotation_steps))


def execute_command(command):
    action, args = parse_command(command)
    write_status("BUSY")

    if action == "move_to_point":
        point = parse_move_args(args)
        if point is not None:
            start_move_to_point(point[0], point[1])
            return

    if action == "rotate":
        angle = parse_rotate_args(args)
        if angle is not None:
            start_rotation(angle)
            return

    if action == "return_to_base":
        start_move_to_point(base_x, base_y)
        return

    if action in ("scan_with_lidar", "detect_with_camera", "wait_for_signal", "communicate_with_apm", "retrieve_object"):
        print("Isaac CyberScape:", action, "simulated complete")
        write_status("DONE")
        return

    print("Isaac CyberScape: unsupported command, marking done:", command)
    write_status("DONE")


def any_busy():
    return active_movement or active_rotation


def check_for_new_command():
    global last_mtime
    if any_busy() or not os.path.exists(COMMAND_FILE):
        return

    current_mtime = os.path.getmtime(COMMAND_FILE)
    if current_mtime == last_mtime:
        return

    last_mtime = current_mtime
    with open(COMMAND_FILE, "r") as file:
        command = file.read().strip()

    if command:
        print("Isaac CyberScape received:", command)
        execute_command(command)


def update_movement():
    global active_movement, last_move_step_time
    if not path_steps:
        if active_movement:
            active_movement = False
            write_status("DONE")
            print("Isaac CyberScape: movement finished")
        return

    now = time.time()
    if now - last_move_step_time < STEP_DELAY:
        return

    last_move_step_time = now
    next_x, next_y = path_steps.pop(0)
    set_robot_position(next_x, next_y)


def update_rotation():
    global active_rotation, last_rotation_step_time
    if not rotation_steps:
        if active_rotation:
            active_rotation = False
            write_status("DONE")
            print("Isaac CyberScape: rotation finished")
        return

    now = time.time()
    if now - last_rotation_step_time < ROTATION_DELAY:
        return

    last_rotation_step_time = now
    set_robot_rotation_z(rotation_steps.pop(0))


def on_update(event):
    check_for_new_command()
    update_movement()
    update_rotation()


if "_cyberscape_isaac_listener" in globals():
    try:
        _cyberscape_isaac_listener.unsubscribe()
    except Exception:
        pass

write_status("DONE")
_cyberscape_isaac_listener = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)

print("Isaac CyberScape listener started for", ROBOT_PATH)
