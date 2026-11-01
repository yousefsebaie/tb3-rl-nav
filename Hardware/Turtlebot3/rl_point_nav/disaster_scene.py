"""Earthquake set dressing for cyberscape_rl_bridge.py --scenario disaster (Isaac Sim GUI only).

Visual props only: no physics, created once, never deleted. They are placed so the
robot's 2D LiDAR (a thin plane ~0.17 m above the floor) sees the same world the policy was
tested in:
  * anything crossing 0.10-0.25 m height stays inside a debris block's footprint
    (the pooled obstacle cubes the RL policy actually avoids), or far from the routes;
  * everything near the routes is lower than 0.10 m (rubble, trapped survivors, tape);
  * tall background props stay >= 0.9 m from the routes or behind the start (the policy
    only uses LiDAR sectors within +-120 deg of the heading).

All positions are in the mission START frame (x ahead, y left, metres; floor = 0).
"""

from __future__ import annotations

import math
import random

CONCRETE = (0.46, 0.44, 0.41)
CONCRETE_DARK = (0.30, 0.29, 0.27)
DUST = (0.55, 0.52, 0.47)
REBAR = (0.35, 0.18, 0.10)
WOOD = (0.40, 0.27, 0.15)
STEEL = (0.25, 0.26, 0.28)
SKIN = (0.80, 0.62, 0.50)
HAIR = (0.15, 0.10, 0.07)
SHIRT = ((0.20, 0.35, 0.65), (0.65, 0.20, 0.18))
TROUSERS = (0.18, 0.20, 0.26)
HAZARD_YELLOW = (0.95, 0.80, 0.05)
HAZARD_BLACK = (0.05, 0.05, 0.05)
TARP = (0.95, 0.45, 0.05)
ROOT = "/World/rl_disaster"

# Corridors the robot drives (start -> A, A -> B including the detour around the middle
# block). Low rubble is kept out of them so the robot never visibly drives through props.
ROUTES = (((0.0, 0.0), (1.5, 0.75)), ((1.5, 0.75), (1.5, -0.75)))
ROUTE_CLEARANCE_M = 0.40


class _Builder:
    def __init__(self, env, to_world, yaw0: float):
        from pxr import UsdGeom

        self.env, self.to_world, self.yaw0 = env, to_world, yaw0
        self.stage, self.units = env._stage, env._meters_per_unit
        self.floor = env._ground_z
        self.count = 0
        UsdGeom.Xform.Define(self.stage, ROOT)

    def _place(self, prim, x, y, z, rot_deg=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        from pxr import Gf, UsdGeom

        wx, wy = self.to_world(x, y)
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(self.env._placement_xyz(wx, wy, self.floor + z))
        api.SetRotate(Gf.Vec3f(rot_deg[0], rot_deg[1], rot_deg[2] + math.degrees(self.yaw0)))
        api.SetScale(Gf.Vec3f(*(s / self.units for s in scale)) if scale is not None else Gf.Vec3f(1.0, 1.0, 1.0))

    def _path(self, name: str) -> str:
        self.count += 1
        return f"{ROOT}/{name}_{self.count}"

    def box(self, name, x, y, z, size, color, rot=(0.0, 0.0, 0.0)):
        """Box of size (sx, sy, sz) metres centred at (x, y, z)."""
        from pxr import Gf, UsdGeom

        cube = UsdGeom.Cube.Define(self.stage, self._path(name))
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        self._place(cube.GetPrim(), x, y, z, rot, size)
        return cube

    def sphere(self, name, x, y, z, radius, color, squash=(1.0, 1.0, 1.0)):
        from pxr import Gf, UsdGeom

        sphere = UsdGeom.Sphere.Define(self.stage, self._path(name))
        sphere.CreateRadiusAttr(1.0)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        self._place(sphere.GetPrim(), x, y, z, scale=tuple(radius * s for s in squash))
        return sphere

    def cylinder(self, name, x, y, z, radius, length, color, rot=(0.0, 0.0, 0.0)):
        """Cylinder along its local z, centred at (x, y, z)."""
        from pxr import Gf, UsdGeom

        cyl = UsdGeom.Cylinder.Define(self.stage, self._path(name))
        cyl.CreateRadiusAttr(1.0)
        cyl.CreateHeightAttr(1.0)
        cyl.CreateAxisAttr(UsdGeom.Tokens.z)
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        self._place(cyl.GetPrim(), x, y, z, rot, (radius, radius, length))
        return cyl

    def capsule(self, name, x, y, z, radius, length, color, rot=(0.0, 0.0, 0.0)):
        """Capsule along its local x (rounded limb / torso), centred at (x, y, z)."""
        from pxr import Gf, UsdGeom

        cap = UsdGeom.Capsule.Define(self.stage, self._path(name))
        cap.CreateRadiusAttr(radius / self.units)
        cap.CreateHeightAttr(max(length - 2 * radius, 0.001) / self.units)
        cap.CreateAxisAttr(UsdGeom.Tokens.x)
        cap.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        self._place(cap.GetPrim(), x, y, z, rot, None)
        return cap

    def light(self, name, x, y, z, color, intensity, radius=0.08):
        from pxr import Gf, UsdLux

        light = UsdLux.SphereLight.Define(self.stage, self._path(name))
        light.CreateRadiusAttr(radius / self.units)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(*color))
        self._place(light.GetPrim(), x, y, z)
        return light


def _near_route(x: float, y: float, margin: float) -> bool:
    for (ax, ay), (bx, by) in ROUTES:
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
        if math.hypot(x - (ax + t * dx), y - (ay + t * dy)) < margin:
            return True
    return math.hypot(x, y) < margin or (1.15 < x < 2.05 and abs(y) < 0.8)


def _debris_block(b: _Builder, x: float, y: float, width: float, rng: random.Random) -> None:
    """Dress one RL obstacle cube (0.5 m tall) as a collapsed concrete pile.

    Overhangs only above 0.27 m and the rubble skirt only below 0.09 m, so the LiDAR
    plane (~0.17 m) still sees exactly the cube the policy avoids.
    """
    half = width / 2
    # Broken slabs piled on top and sliding off, overhanging (all above 0.27 m).
    b.box("slab", x + rng.uniform(-0.04, 0.04), y + rng.uniform(-0.04, 0.04), 0.53,
          (width * 1.5, width * 1.0, 0.05), CONCRETE_DARK, (rng.uniform(-6, 6), rng.uniform(-8, 8), rng.uniform(0, 40)))
    b.box("slab", x + rng.uniform(-0.06, 0.06), y + rng.uniform(-0.06, 0.06), 0.45,
          (width * 1.35, width * 0.55, 0.05), CONCRETE, (rng.uniform(-4, 4), 28.0, rng.uniform(0, 180)))
    b.box("chunk", x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05), 0.6,
          (width * 0.5, width * 0.4, 0.09), CONCRETE, (rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(0, 90)))
    for _ in range(4):
        b.cylinder("rebar", x + rng.uniform(-half, half) * 0.7, y + rng.uniform(-half, half) * 0.7, 0.66,
                   0.006, 0.34, REBAR, (rng.uniform(-40, 40), rng.uniform(-40, 40), 0.0))
    # Rubble skirt around the foot (below 0.09 m), kept out of the robot's corridors.
    for _ in range(40):
        angle, dist = rng.uniform(0, 2 * math.pi), half + rng.uniform(0.0, 0.28)
        px, py = x + dist * math.cos(angle), y + dist * math.sin(angle)
        if _near_route(px, py, 0.2):
            continue
        size = rng.uniform(0.025, 0.09)
        b.box("rubble", px, py, size * 0.3, (size, size * rng.uniform(0.6, 1.3), size * 0.6),
              rng.choice((CONCRETE, CONCRETE_DARK, DUST)), (rng.uniform(0, 30), rng.uniform(0, 30), rng.uniform(0, 90)))


def _survivor(b: _Builder, x: float, y: float, heading_deg: float, shirt, rng: random.Random) -> None:
    """A person lying face up, legs pinned under a fallen slab, one arm raised for help.

    Everything stays below 0.10 m (under the LiDAR plane). (x, y) is the head.
    """
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))

    def at(along: float, side: float):
        return x + along * c - side * s, y + along * s + side * c

    def limb(along, side, z, radius, length, color, yaw=0.0, pitch=0.0):
        px, py = at(along, side)
        b.capsule("survivor", px, py, z, radius, length, color, (0.0, pitch, heading_deg + yaw))

    hx, hy = at(0.0, 0.0)
    b.sphere("survivor", hx, hy, 0.06, 0.06, SKIN, (1.15, 0.95, 0.9))                  # head
    hx, hy = at(-0.035, 0.0)
    b.sphere("survivor", hx, hy, 0.05, 0.058, HAIR, (1.0, 1.0, 0.75))                   # hair
    limb(0.11, 0.0, 0.04, 0.03, 0.08, SKIN)                                               # neck
    limb(0.33, 0.0, 0.055, 0.085, 0.42, shirt)                                            # chest
    limb(0.33, 0.0, 0.05, 0.075, 0.42, shirt, 0.0, 0.0)
    limb(0.30, 0.20, 0.04, 0.035, 0.30, shirt, -20.0)                                     # left arm, raised
    limb(0.10, 0.27, 0.05, 0.03, 0.26, SKIN, 40.0, -12.0)
    limb(0.30, -0.20, 0.04, 0.035, 0.30, shirt, 15.0)                                     # right arm
    limb(0.52, -0.25, 0.035, 0.03, 0.24, SKIN, 10.0)
    limb(0.70, 0.08, 0.045, 0.06, 0.40, TROUSERS, 3.0)                                    # legs
    limb(0.70, -0.08, 0.045, 0.06, 0.40, TROUSERS, -3.0)
    # Fallen slab pinning the legs, plus broken pieces and dust (all below 0.10 m).
    px, py = at(0.85, 0.0)
    b.box("pin", px, py, 0.085, (0.55, 0.70, 0.045), CONCRETE_DARK, (5.0, -4.0, heading_deg + 20.0))
    for _ in range(14):
        px, py = at(rng.uniform(0.5, 1.25), rng.uniform(-0.45, 0.45))
        size = rng.uniform(0.025, 0.07)
        b.box("rubble", px, py, size * 0.35, (size, size * rng.uniform(0.6, 1.3), size * 0.7),
              rng.choice((CONCRETE, DUST, CONCRETE_DARK)), (rng.uniform(0, 25), rng.uniform(0, 25), rng.uniform(0, 90)))
    for _ in range(10):
        px, py = at(rng.uniform(-0.2, 1.1), rng.uniform(-0.45, 0.45))
        b.box("dust", px, py, 0.002, (rng.uniform(0.08, 0.2), rng.uniform(0.08, 0.2), 0.003), DUST,
              (0.0, 0.0, rng.uniform(0, 90)))


def _collapse_zone(b: _Builder, rng: random.Random) -> None:
    """Fallen ceiling and furniture ahead of the search area (>= 0.9 m from the routes)."""
    b.box("ceiling", 2.95, 0.2, 0.35, (1.3, 0.9, 0.06), CONCRETE, (0, 24, 12))
    b.box("ceiling", 3.1, -0.6, 0.22, (1.0, 0.7, 0.06), CONCRETE_DARK, (0, -16, -25))
    b.cylinder("beam", 3.2, 1.2, 0.3, 0.07, 1.6, STEEL, (0, 75, 30))
    b.box("shelf", 2.7, -1.0, 0.12, (0.9, 0.35, 0.24), WOOD, (0, 0, 70))
    b.box("ceiling", -0.2, 2.3, 0.3, (1.4, 0.8, 0.06), CONCRETE, (20, 0, -8))
    b.cylinder("beam", 0.4, 2.7, 0.25, 0.06, 1.8, STEEL, (78, 0, 10))
    for _ in range(45):
        px, py = rng.uniform(2.3, 3.6), rng.uniform(-1.1, 1.4)
        if rng.random() < 0.5:
            px, py = rng.uniform(-0.8, 1.2), rng.uniform(1.6, 2.8)
        s = rng.uniform(0.04, 0.16)
        b.box("rubble", px, py, s * 0.35, (s, s * rng.uniform(0.5, 1.3), s * 0.6),
              rng.choice((CONCRETE, CONCRETE_DARK, DUST)), (rng.uniform(0, 25), rng.uniform(0, 25), rng.uniform(0, 90)))


def _responder_base(b: _Builder) -> None:
    """Triage tarp, hazard tape and an emergency light behind the robot's start."""
    b.box("tarp", -0.65, 0.0, 0.003, (0.6, 1.0, 0.004), TARP)
    for i in range(14):   # tape along the floor at the edge of the search area
        b.box("tape", -0.3, -0.95 + i * 0.14, 0.004, (0.03, 0.12, 0.004),
              HAZARD_YELLOW if i % 2 == 0 else HAZARD_BLACK)
    for side in (-1.0, 1.0):
        b.cylinder("post", -0.9, side * 0.6, 0.45, 0.02, 0.9, HAZARD_YELLOW)
    b.box("tape", -0.9, 0.0, 0.8, (0.01, 1.2, 0.04), HAZARD_YELLOW)
    b.cylinder("mast", -0.95, 0.0, 0.7, 0.025, 1.4, STEEL)
    b.box("lamp", -0.95, 0.0, 1.42, (0.12, 0.25, 0.08), STEEL)


def _lighting(b: _Builder) -> None:
    """Dim the room, add a cold emergency flood, a red beacon and a warm fire glow."""
    from pxr import Usd, UsdLux

    for prim in Usd.PrimRange(b.stage.GetPseudoRoot()):
        if prim.HasAPI(UsdLux.LightAPI) and not str(prim.GetPath()).startswith(ROOT):
            light = UsdLux.LightAPI(prim)
            value = light.GetIntensityAttr().Get()
            if value:
                light.GetIntensityAttr().Set(value * 0.35)
    b.light("flood", -0.9, 0.0, 1.35, (0.85, 0.92, 1.0), 9.0e4, 0.06)
    b.light("beacon", -0.9, 0.6, 0.95, (1.0, 0.05, 0.02), 3.0e4, 0.03)
    b.light("fire", 3.0, 0.1, 0.25, (1.0, 0.45, 0.12), 6.0e4, 0.15)
    b.light("fire", -0.1, 2.4, 0.2, (1.0, 0.5, 0.15), 3.0e4, 0.12)


def build(env, to_world, yaw0: float, debris_blocks, survivors) -> None:
    """debris_blocks: (x, y, width) of the RL obstacle cubes; survivors: checkpoint (x, y)."""
    from pxr import Gf, UsdGeom

    rng = random.Random(7)
    b = _Builder(env, to_world, yaw0)
    for path in env._obstacle_paths:
        UsdGeom.Gprim(env._stage.GetPrimAtPath(path)).CreateDisplayColorAttr([Gf.Vec3f(*CONCRETE)])
    for x, y, width in debris_blocks:
        _debris_block(b, x, y, width, rng)
    # Each survivor lies just beyond its checkpoint, pointing away from the robot's approach.
    for index, (x, y) in enumerate(survivors):
        side = 1.0 if y >= 0 else -1.0
        heading = 25.0 if side > 0 else -5.0   # right side: clear of the chairs at y < -1.3
        _survivor(b, x + 0.25, y + side * 0.3, heading, SHIRT[index % len(SHIRT)], rng)
    _collapse_zone(b, rng)
    _responder_base(b)
    _lighting(b)
