#!/usr/bin/env python3
"""One-process LiDAR-only integration test. No policy or obstacles are created."""
import argparse
import numpy as np

try:
    from .config import PointNavAction
    from .isaac_point_nav_lidar_minimal_env import IsaacPointNavLidarMinimalEnv, RANGE_SETTINGS
except ImportError:
    from config import PointNavAction
    from isaac_point_nav_lidar_minimal_env import IsaacPointNavLidarMinimalEnv, RANGE_SETTINGS


def verify(env, label, obs, info):
    print(f"[TB03_MINIMAL_TEST] OBS_{label}={obs.tolist()}", flush=True)
    assert obs.shape == (6,) and obs.dtype == np.float32
    assert np.isfinite(obs).all() and env.observation_space.contains(obs)
    assert ((obs[3:] >= 0) & (obs[3:] <= 1)).all()
    debug = info["lidar"]
    assert debug["parent"] == "/World"
    assert np.allclose(debug["world_scale"], [1, 1, 1], atol=1e-6, rtol=0)
    assert np.allclose(debug["relative_offset_m"], [0.10, 0, 0.22], atol=1e-4, rtol=0)
    assert debug["orientation_aligned"] and debug["follow_offset_error_max_m"] < 1e-4
    for name, expected in RANGE_SETTINGS.items():
        assert np.isclose(debug["settings"][name], expected, atol=1e-6, rtol=0)
    scan = debug["scan"]
    assert scan is not None and scan["num_cols"] > 0 and scan["samples"] > 0
    assert scan["finite_valid"] > 0, "No real finite returns anywhere in FlatScan"
    assert scan["samples"] == scan["num_cols"] == len(scan["ranges_m"])
    assert np.allclose(scan["depth_range_m"], [0.10, 200], atol=1e-5, rtol=0)
    az = (np.linspace(*scan["azimuth_range_deg"], scan["num_cols"]) + 180) % 360 - 180
    np.testing.assert_allclose(az, scan["azimuth_deg"])
    ranges = scan["ranges_m"]
    valid = np.isfinite(ranges) & (ranges > 1e-3) & (ranges >= scan["depth_range_m"][0]) & (ranges <= scan["depth_range_m"][1])
    for i, (name, lo, hi) in enumerate((("front", -30, 30), ("left", 30, 120), ("right", -120, -30))):
        mask = valid & (az >= lo) & (az <= hi)
        minimum = float(ranges[mask].min()) if mask.any() else None
        expected = 1.0 if minimum is None else np.clip(minimum / env.clearance_max_m, 0, 1)
        sector = info["lidar_sectors"][name]
        assert sector["valid_rays"] == int(mask.sum())
        assert sector["raw_min_m"] == minimum
        assert sector["no_valid_return"] == (minimum is None)
        np.testing.assert_allclose(obs[3 + i], expected, atol=1e-6)
        print(f"[TB03_MINIMAL_TEST] {name.upper()}_RAW_M={minimum} "
              f"{name.upper()}_NORM={obs[3 + i]:.6f} VALID_RAYS_{name.upper()}={int(mask.sum())}", flush=True)
    print(f"[TB03_MINIMAL_TEST] {label}=PASS cols={scan['num_cols']} "
          f"finite_valid={scan['finite_valid']} offset_error_max={debug['follow_offset_error_max_m']:.9f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Default uses the validated visible renderer")
    args = parser.parse_args()
    env = IsaacPointNavLidarMinimalEnv(headless=args.headless)
    try:
        obs, info = env.reset(seed=42, options={"goal": (-2.118, 0.444)})
        verify(env, "RESET", obs, info)
        assert info["episode_steps"] == 0 and info["episode_elapsed_sim_s"] == 0
        elapsed = 0.0
        for index, (label, action) in enumerate((("FORWARD", PointNavAction.FORWARD),
                                                ("LEFT", PointNavAction.TURN_LEFT),
                                                ("RIGHT", PointNavAction.TURN_RIGHT)), 1):
            obs, reward, terminated, truncated, info = env.step(action)
            verify(env, label, obs, info)
            _, frames = env._action_spec(action)
            elapsed += (frames + env.isaac_config.settle_frames) * env._physics_dt_s
            assert np.isclose(info["episode_elapsed_sim_s"], elapsed)
            assert info["episode_steps"] == index
            cfg = env.point_nav_config
            expected_reward = cfg.progress_reward_scale * np.clip(info["progress_m"], -cfg.progress_clip_m, cfg.progress_clip_m) - cfg.step_penalty
            if terminated:
                expected_reward += cfg.success_bonus
            assert np.isclose(reward, expected_reward)
            assert not terminated and not truncated, "Episode ended before the required smoke sequence completed"
        assert env.isaac_config.goal_tolerance_m == 0.05
        print("[TB03_MINIMAL_TEST] OVERALL_PASS=PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
