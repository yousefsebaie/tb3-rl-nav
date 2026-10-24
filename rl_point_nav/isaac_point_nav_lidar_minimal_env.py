"""Minimal LiDAR observation layer over unchanged V1B point navigation.

No obstacle creation, collision logic, visual hiding, reward changes or training.
RTX scans accumulate over time; capture_update is a polling time, not an exact
per-ray measurement timestamp. Actions never wait additional physics frames.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from gymnasium import spaces

try:
    from .isaac_point_nav_env import IsaacPointNavEnv, ARTICULATION_PATH
except ImportError:
    from isaac_point_nav_env import IsaacPointNavEnv, ARTICULATION_PATH

BASE_LINK_PATH = f"{ARTICULATION_PATH}/a__namespace_base_link"
LIDAR_PATH = "/World/tb03_minimal_runtime_lidar"
LIDAR_OFFSET = (0.10, 0.0, 0.22)
LIDAR_CLEARANCE_MAX_M = 1.0
FLAT = "IsaacComputeRTXLidarFlatScan"
RAW = "IsaacExtractRTXSensorPointCloudNoAccumulator"
RANGE_SETTINGS = {
    "omni:sensor:Core:nearRangeM": 0.10,
    "omni:sensor:Core:minDistBetweenEchosM": 0.10,
    "omni:sensor:Core:farRangeM": 200.0,
}
SECTORS = {"front": (-30.0, 30.0), "left": (30.0, 120.0), "right": (-120.0, -30.0)}


class _RenderedLidarContext:
    """Intercept both V1B action steps and settling steps, without adding steps."""

    def __init__(self, context, env):
        self.context, self.env = context, env

    def step(self, render=True):
        self.env._sync_lidar()
        self.context.step(render=True)
        self.env._sync_lidar()
        self.env._lidar_updates += 1
        self.env._capture_scan()

    def __getattr__(self, name):
        return getattr(self.context, name)


class IsaacPointNavLidarMinimalEnv(IsaacPointNavEnv):
    """V1B navigation with three additional normalized LiDAR clearances."""

    def __init__(self, *args, clearance_max_m=LIDAR_CLEARANCE_MAX_M, **kwargs):
        self.clearance_max_m = float(clearance_max_m)
        if not math.isfinite(self.clearance_max_m) or self.clearance_max_m <= 0:
            raise ValueError("clearance_max_m must be finite and positive")
        self._lidar = None
        self._lidar_stage = None
        self._owns_lidar_prim = False
        self._flat_annotator = None
        self._scan = None
        self._lidar_updates = 0
        self._follow_error_max_m = 0.0
        self._mount_info = {}
        # One SimulationApp only. V1B initialization and configuration unchanged.
        super().__init__(*args, **kwargs)
        try:
            self._create_lidar()
            self.observation_space = spaces.Box(
                low=np.array([0, -1, -1, 0, 0, 0], dtype=np.float32),
                high=np.ones(6, dtype=np.float32),
                dtype=np.float32,
            )
        except Exception:
            self.close()
            raise

    def _create_lidar(self):
        import omni.usd
        from isaacsim.sensors.rtx import LidarRtx

        self._lidar_stage = omni.usd.get_context().get_stage()
        if self._lidar_stage.GetPrimAtPath(LIDAR_PATH).IsValid():
            raise RuntimeError(f"Runtime sensor path already exists: {LIDAR_PATH}")
        if not self._lidar_stage.GetPrimAtPath(BASE_LINK_PATH).IsValid():
            raise RuntimeError(f"Missing base link: {BASE_LINK_PATH}")
        self._owns_lidar_prim = True
        self._lidar = LidarRtx(
            prim_path=LIDAR_PATH, name="tb03_minimal_runtime_lidar",
            config_file_name="Example_Rotary_2D", **RANGE_SETTINGS,
        )
        self._lidar.initialize()
        self._lidar.set_local_scale(np.ones(3))
        self._sync_lidar()
        self._check_live_ranges()
        if not self._lidar.get_render_product_path():
            raise RuntimeError("RTX render product is missing")
        self._lidar.attach_annotator(RAW)
        self._lidar.attach_annotator(FLAT)
        self._flat_annotator = self._lidar.get_annotators()[FLAT]
        self._simulation_context = _RenderedLidarContext(self._simulation_context, self)
        print(f"[TB03_MINIMAL] Created {LIDAR_PATH}; offset={LIDAR_OFFSET}", flush=True)

    def _check_live_ranges(self):
        prim = self._lidar_stage.GetPrimAtPath(LIDAR_PATH)
        result = {}
        for name, expected in RANGE_SETTINGS.items():
            value = prim.GetAttribute(name).Get()
            if value is None or not np.isclose(value, expected, atol=1e-6, rtol=0):
                raise RuntimeError(f"LiDAR setting mismatch: {name}={value}")
            result[name] = float(value)
        return result

    def _sync_lidar(self):
        from pxr import Gf, UsdGeom

        stage = self._lidar_stage
        base = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(BASE_LINK_PATH))
        ok, _, _, rotation_matrix, translation, _ = base.Factor()
        if not ok:
            raise RuntimeError("Cannot decompose base-link transform")
        rotation = rotation_matrix.ExtractRotationQuat().GetNormalized()
        units = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(units) or units <= 0:
            raise RuntimeError("Invalid stage units")
        offset = Gf.Vec3d(*(v / units for v in LIDAR_OFFSET))
        position = translation + Gf.Rotation(rotation).TransformDir(offset)
        quat = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=float)
        self._lidar.set_world_pose(position=np.array(position), orientation=quat)
        prim = stage.GetPrimAtPath(LIDAR_PATH)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        ok, _, scale, sensor_rotation, measured_position, _ = world.Factor()
        relative = Gf.Rotation(rotation).GetInverse().TransformDir(measured_position - translation)
        relative_m = np.array(relative) * units
        error = float(np.linalg.norm(relative_m - LIDAR_OFFSET))
        sq = sensor_rotation.ExtractRotationQuat().GetNormalized()
        sensor_quat = np.array([sq.GetReal(), *sq.GetImaginary()], dtype=float)
        aligned = bool(abs(float(np.dot(quat, sensor_quat))) >= 1 - 1e-6)
        scale_array = np.array(scale, dtype=float)
        if not ok or not np.allclose(scale_array, 1, atol=1e-6, rtol=0) or not aligned:
            raise RuntimeError("LiDAR must have unit world scale and base-aligned rotation")
        if error > 1e-4 or str(prim.GetParent().GetPath()) != "/World":
            raise RuntimeError("LiDAR attachment invariant failed")
        self._follow_error_max_m = max(self._follow_error_max_m, error)
        self._mount_info = {
            "path": LIDAR_PATH, "parent": str(prim.GetParent().GetPath()),
            "world_scale": scale_array.tolist(), "relative_offset_m": relative_m.tolist(),
            "orientation_aligned": aligned, "follow_offset_error_max_m": self._follow_error_max_m,
        }

    def _capture_scan(self):
        """Keep the latest complete scan, never the most-obstructed/best scan."""
        data = self._flat_annotator.get_data()
        if not isinstance(data, dict):
            return
        ranges = np.asarray(data.get("linearDepthData", []), dtype=float).reshape(-1)
        cols = int(data.get("numCols", 0))
        bounds = np.asarray(data.get("azimuthRange", []), dtype=float).reshape(-1)
        depth = np.asarray(data.get("depthRange", []), dtype=float).reshape(-1)
        if cols <= 0 or ranges.size != cols or bounds.size != 2 or not np.isfinite(bounds).all():
            return
        if depth.size != 2 or not np.allclose(depth, [0.10, 200.0], atol=1e-5, rtol=0):
            raise RuntimeError(f"Unexpected FlatScan depthRange: {depth}")
        # Installed Isaac 5.1 RTX tests reconstruct azimuth from these endpoints.
        angles = (np.linspace(bounds[0], bounds[1], cols) + 180) % 360 - 180
        valid = np.isfinite(ranges) & (ranges > 1e-3) & (ranges >= depth[0]) & (ranges <= depth[1])
        self._scan = {
            "ranges_m": ranges.copy(), "azimuth_deg": angles, "valid": valid,
            "azimuth_range_deg": bounds.copy(), "depth_range_m": depth.copy(),
            "num_cols": cols, "samples": int(ranges.size), "finite_valid": int(valid.sum()),
            "capture_update": self._lidar_updates,
        }

    def _reset_articulation(self):
        # Preserve V1B's proven reset first, then acquire a new stationary scan.
        super()._reset_articulation()
        self._lidar.detach_annotator(FLAT)
        self._lidar.attach_annotator(FLAT)
        self._flat_annotator = self._lidar.get_annotators()[FLAT]
        self._scan = None
        self._follow_error_max_m = 0.0
        # Warm-up belongs to reset, not the episode. No extra frames in step().
        for frame in range(360):
            self._step_frames(1, count_episode_time=False, hold_stop=True)
            if frame >= 179 and self._scan is not None:
                print(f"[TB03_MINIMAL] Reset scan ready after {frame + 1} rendered frames", flush=True)
                return self._state()
        raise RuntimeError("No complete FlatScan during bounded reset warm-up")

    def step(self, action):
        self._scan = None  # Never silently reuse a scan cached before this action.
        try:
            return super().step(action)
        except Exception:
            self._apply_stop()
            self._episode_active = False
            raise

    def _sector_readings(self):
        if self._scan is None:
            raise RuntimeError("No complete FlatScan available in this reset/action")
        angles, ranges, valid = (self._scan[k] for k in ("azimuth_deg", "ranges_m", "valid"))
        readings = {}
        for name, (lo, hi) in SECTORS.items():
            selected = valid & (angles >= lo) & (angles <= hi)
            count = int(selected.sum())
            minimum = float(ranges[selected].min()) if count else None
            readings[name] = {
                "raw_min_m": minimum, "valid_rays": count, "no_valid_return": count == 0,
                "normalized": 1.0 if minimum is None else float(np.clip(minimum / self.clearance_max_m, 0, 1)),
            }
        return readings

    def _observation(self, state):
        sectors = self._sector_readings()
        obs = np.concatenate((super()._observation(state),
                              np.array([sectors[k]["normalized"] for k in SECTORS], dtype=np.float32)))
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Invalid minimal observation: {obs}")
        return obs

    def _info(self, state, **kwargs):
        info = super()._info(state, **kwargs)
        info["lidar_sectors"] = self._sector_readings()
        info["lidar"] = self.lidar_debug()
        return info

    def lidar_debug(self):
        """Copies of actual scan data allow independent smoke-test verification."""
        self._sync_lidar()
        result = deepcopy(self._mount_info)
        result["settings"] = self._check_live_ranges()
        result["scan"] = deepcopy(self._scan)
        result["capture_age_updates"] = None if self._scan is None else (
            self._lidar_updates - self._scan["capture_update"])
        result["scan_time_note"] = "Latest available accumulated scan; capture time is not per-ray time"
        return result

    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            self._apply_stop()
            if self._lidar is not None:
                self._lidar.pause()
                for name, method in (("_acquisition_callback", "reset"),
                                     ("_stage_open_callback", "reset"),
                                     ("_timer_reset_callback", "unsubscribe")):
                    callback = getattr(self._lidar, name, None)
                    if callback is not None:
                        getattr(callback, method)()
                        setattr(self._lidar, name, None)
                self._lidar.detach_all_annotators()
                self._lidar.detach_all_writers()
                product = getattr(self._lidar, "_render_product", None)
                if product is not None:
                    product.destroy()
                    self._lidar._render_product = None
                self._lidar = None
            if self._owns_lidar_prim and self._lidar_stage is not None:
                self._lidar_stage.RemovePrim(LIDAR_PATH)
        finally:
            super().close()
