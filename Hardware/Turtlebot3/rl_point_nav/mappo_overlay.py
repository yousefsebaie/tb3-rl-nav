"""GUI-only overlay for IsaacMultiPointNavEnv: draws the hidden robots in the viewport.

Training hides the robot meshes because the RTX LiDAR would see them. The debug-draw overlay
is drawn by the viewport, not added to the USD stage, so the LiDAR (and the policy) see
exactly what they saw in training. Per robot, in its goal colour:
  footprint circle (0.105 m), heading line, trail, dashed line to its goal;
  a red X where a robot collided, a grey ring when it timed out.
"""

from __future__ import annotations

import math

COLLISION_RGBA = (1.0, 0.1, 0.1, 1.0)
TIMEOUT_RGBA = (0.6, 0.6, 0.6, 1.0)
DRAW_HEIGHT_M = 0.12   # above the floor, below the obstacle tops


class MultiRobotOverlay:
    def __init__(self, env) -> None:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw

        self._draw = _debug_draw.acquire_debug_draw_interface()
        self._env = env
        self._unit = 1.0 / env._meters_per_unit
        self._z = getattr(env, "_ground_z", 0.0) + DRAW_HEIGHT_M * self._unit
        self._trails: dict[int, list[tuple]] = {}
        self._colors = []
        from isaac_multi_point_nav_env import GOAL_COLORS
        self._colors = [(*c, 1.0) for c in GOAL_COLORS]

    def _p(self, x: float, y: float) -> tuple:
        return (x * self._unit, y * self._unit, self._z)

    def frame_camera(self, info: dict | None = None) -> None:
        """Camera below the room's ceiling (above it the scene is unlit).

        VIEW=top (default): near-overhead, centred on this episode's starts and goals (parked robot left out).
        VIEW=angled: the fixed angled view over the whole encounter area.
        """
        import os
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view

        mc = self._env.multi_config
        if os.environ.get("VIEW", "top") == "angled" or info is None:
            cx, cy = sum(mc.area_x_m) / 2, sum(mc.area_y_m) / 2
            eye = np.array(self._p(cx, cy - 2.6))
            eye[2] = self._z + 2.1 * self._unit
        else:
            robots = info["robots"][:1] if mc.park_teammate else info["robots"]
            xs = [p for r in robots for p in (r["pose"]["x"], r["goal"]["x"])]
            ys = [p for r in robots for p in (r["pose"]["y"], r["goal"]["y"])]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            eye = np.array(self._p(cx, cy - 0.12))   # slight tilt keeps the up vector well defined
            eye[2] = self._z + min(1.1 + 0.55 * span, 2.1) * self._unit
        set_camera_view(eye=eye, target=np.array(self._p(cx, cy)))

    def frame_overhead(self, x_range: tuple, y_range: tuple, height_m: float = 2.0) -> None:
        """Straight-down view of a whole box. The ceiling caps the height, so the lens is widened
        (shorter focal length) until the box's y extent fits the viewport."""
        import numpy as np
        import omni.usd
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Usd, UsdGeom

        cx, cy = sum(x_range) / 2, sum(y_range) / 2
        eye = np.array(self._p(cx, cy - 0.05))   # slight tilt keeps the up vector well defined
        eye[2] = self._z + height_m * self._unit
        set_camera_view(eye=eye, target=np.array(self._p(cx, cy)))
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera(stage.GetPrimAtPath("/OmniverseKit_Persp"))
        if camera:
            aperture = camera.GetHorizontalApertureAttr().Get() or 20.955
            half = 0.5 * max(y_range[1] - y_range[0], (x_range[1] - x_range[0]) / 1.7)
            focal = float(aperture / 1.7 / 2 * (height_m + 0.12) / half)
            with Usd.EditContext(stage, stage.GetSessionLayer()):   # the viewport camera lives there
                camera.GetFocalLengthAttr().Set(focal)
            print(f"[TB03_MAPPO] overhead camera focal={camera.GetFocalLengthAttr().Get():.1f} mm", flush=True)

    def new_episode(self) -> None:
        self._trails = {}

    def update(self, info: dict) -> None:
        d = self._draw
        d.clear_lines()
        d.clear_points()
        starts, ends, colors, widths = [], [], [], []

        def seg(a, b, color, width):
            starts.append(a); ends.append(b); colors.append(color); widths.append(width)

        for i, robot in enumerate(info["robots"]):
            color = self._colors[i % len(self._colors)]
            x, y, yaw = robot["pose"]["x"], robot["pose"]["y"], robot["yaw"]
            self._trails.setdefault(i, []).append(self._p(x, y))
            r = 0.105
            ring = [self._p(x + r * math.cos(k * math.pi / 12), y + r * math.sin(k * math.pi / 12)) for k in range(25)]
            for a, b in zip(ring, ring[1:]):
                seg(a, b, color, 4)
            seg(self._p(x, y), self._p(x + 0.22 * math.cos(yaw), y + 0.22 * math.sin(yaw)), color, 5)
            gx, gy = robot["goal"]["x"], robot["goal"]["y"]
            n = max(int(math.hypot(gx - x, gy - y) / 0.08), 1)
            for k in range(0, n, 2):   # dashed line to the goal
                t0, t1 = k / n, min(k + 1, n) / n
                seg(self._p(x + (gx - x) * t0, y + (gy - y) * t0), self._p(x + (gx - x) * t1, y + (gy - y) * t1),
                    (*color[:3], 0.45), 1)
            trail = self._trails[i]
            for a, b in zip(trail, trail[1:]):
                seg(a, b, (*color[:3], 0.7), 2)
            if robot["outcome"] == "COLLISION":
                s = 0.15
                seg(self._p(x - s, y - s), self._p(x + s, y + s), COLLISION_RGBA, 6)
                seg(self._p(x - s, y + s), self._p(x + s, y - s), COLLISION_RGBA, 6)
            elif robot["outcome"] == "TIMEOUT":
                big = [self._p(x + 0.2 * math.cos(k * math.pi / 12), y + 0.2 * math.sin(k * math.pi / 12))
                       for k in range(25)]
                for a, b in zip(big, big[1:]):
                    seg(a, b, TIMEOUT_RGBA, 3)
        d.draw_lines(starts, ends, colors, widths)
