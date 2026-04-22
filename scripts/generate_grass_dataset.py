#!/usr/bin/env python3
"""Generate grass segmentation clips from Isaac Sim.

Run this with Isaac Sim's Python, not with system Python.
The script creates procedural MVP scenes, captures RGB/semantic/depth frames,
and writes per-frame metadata plus optional MP4 previews.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAMERA_PATH = "/World/Camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Isaac Sim grass segmentation clip dataset."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "dataset_config.yaml"),
        help="Dataset config path. YAML is recommended; JSON also works.",
    )
    parser.add_argument(
        "--label-map",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "label_map.yaml"),
        help="Class label map path. YAML is recommended; JSON also works.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override dataset.output_dir from the config.",
    )
    parser.add_argument(
        "--clips-per-scene",
        type=int,
        default=None,
        help="Override capture.clips_per_scene. Useful for smoke tests.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=None,
        help="Override capture.frames_per_clip. Useful for smoke tests.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless rendering.",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Force visible Isaac Sim UI.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable MP4 preview creation.",
    )
    return parser.parse_args()


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is required for YAML configs. Install PyYAML in the "
                "Isaac Sim Python environment or pass a JSON file."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping/object at the top level.")
    return data


def deep_update(base: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = base
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def get_range(spec: Any, rng: random.Random, default: float = 0.0) -> float:
    if isinstance(spec, dict):
        return rng.uniform(float(spec.get("min", default)), float(spec.get("max", default)))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return rng.uniform(float(spec[0]), float(spec[1]))
    if spec is None:
        return default
    return float(spec)


def get_int_range(spec: Any, rng: random.Random, default: int = 0) -> int:
    if isinstance(spec, dict):
        return rng.randint(int(spec.get("min", default)), int(spec.get("max", default)))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return rng.randint(int(spec[0]), int(spec[1]))
    if spec is None:
        return default
    return int(spec)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sanitize_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in {"_", "-"} else "_")
    return "".join(keep)


@dataclass(frozen=True)
class CameraPose:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    heading_deg: float
    pitch_deg: float
    roll_deg: float


class IsaacGrassDatasetGenerator:
    def __init__(
        self,
        simulation_app: Any,
        cfg: dict[str, Any],
        label_map: dict[str, int],
    ) -> None:
        self.app = simulation_app
        self.cfg = cfg
        self.label_map = {str(k).lower(): int(v) for k, v in label_map.items()}
        self.seed = int(cfg.get("dataset", {}).get("seed", 42))
        self.rng = random.Random(self.seed)

        # These imports must happen after SimulationApp is created.
        import numpy as np  # type: ignore
        import omni.replicator.core as rep  # type: ignore
        import omni.usd  # type: ignore
        import omni.kit.commands  # type: ignore
        from pxr import Gf, Sdf, Semantics, UsdGeom, UsdLux, UsdShade  # type: ignore

        self.np = np
        self.rep = rep
        self.omni_usd = omni.usd
        self.omni_kit_commands = omni.kit.commands
        self.Gf = Gf
        self.Sdf = Sdf
        self.Semantics = Semantics
        self.UsdGeom = UsdGeom
        self.UsdLux = UsdLux
        self.UsdShade = UsdShade
        self._nucleus_root: str | None = None

        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Pillow is required to write PNG files. Install pillow in the "
                "Isaac Sim Python environment."
            ) from exc
        self.Image = Image

    # ------------------------------------------------------------------
    # NVIDIA Assets helpers
    # ------------------------------------------------------------------

    def _get_nucleus_root(self) -> str:
        """Return the Nucleus/CDN root URL, caching the result."""
        if self._nucleus_root is not None:
            return self._nucleus_root
        configured = self.cfg.get("assets", {}).get("nucleus_root", "")
        if configured:
            self._nucleus_root = configured.rstrip("/")
            return self._nucleus_root
        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path  # type: ignore
            root = get_assets_root_path()
        except ImportError:
            try:
                from omni.isaac.core.utils.nucleus import get_assets_root_path  # type: ignore
                root = get_assets_root_path()
            except ImportError:
                root = "omniverse://localhost"
        self._nucleus_root = (root or "omniverse://localhost").rstrip("/")
        print(f"[Assets] Nucleus root: {self._nucleus_root}")
        return self._nucleus_root

    def _load_asset(
        self,
        stage: Any,
        prim_path: str,
        relative_url: str,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        label: str = "",
        instanceable: bool = True,
    ) -> Any | None:
        """Load a USD asset via reference and return its prim, or None on failure."""
        asset_url = f"{self._get_nucleus_root()}/{relative_url}"
        try:
            self.omni_kit_commands.execute(
                "CreateReferenceCommand",
                usd_context=self.omni_usd.get_context(),
                path_to=prim_path,
                asset_path=asset_url,
                instanceable=instanceable,
            )
        except Exception as exc:
            print(f"[Assets] Failed to load {asset_url}: {exc}")
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[Assets] Prim not valid after load: {prim_path}")
            return None
        xform = self.UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(self.UsdGeom.XformOp.PrecisionDouble).Set(self.Gf.Vec3d(*translate))
        xform.AddRotateXYZOp(self.UsdGeom.XformOp.PrecisionDouble).Set(self.Gf.Vec3d(*rotate_xyz))
        xform.AddScaleOp(self.UsdGeom.XformOp.PrecisionDouble).Set(self.Gf.Vec3d(scale, scale, scale))
        if label:
            self._set_semantic_label(prim, label)
        return prim

    def run(self) -> None:
        output_dir = Path(self.cfg["dataset"]["output_dir"]).resolve()
        ensure_dir(output_dir)

        jobs = self._build_clip_jobs()
        manifest: dict[str, Any] = {
            "seed": self.seed,
            "label_map": self.label_map,
            "config": self.cfg,
            "clips": [],
        }

        for clip_idx, job in enumerate(jobs, start=1):
            split = job["split"]
            scene_name = job["scene"]
            clip_id = f"clip_{clip_idx:06d}"
            clip_dir = output_dir / split / clip_id
            ensure_dir(clip_dir)
            print(f"[{clip_idx:04d}/{len(jobs):04d}] {split}/{clip_id} {scene_name}")

            clip_rng = random.Random(self.seed + clip_idx * 1009)
            clip_record = self._generate_clip(
                clip_dir=clip_dir,
                clip_id=clip_id,
                split=split,
                scene_name=scene_name,
                rng=clip_rng,
            )
            manifest["clips"].append(clip_record)

        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Done. Dataset written to: {output_dir}")

    def _build_clip_jobs(self) -> list[dict[str, Any]]:
        scenes = self.cfg.get("scenes", {})
        if not scenes:
            raise ValueError("configs/dataset_config.yaml must define at least one scene.")

        clips_per_scene = int(self.cfg["capture"].get("clips_per_scene", 20))
        jobs: list[dict[str, Any]] = []
        for scene_name in scenes.keys():
            for _ in range(clips_per_scene):
                jobs.append({"scene": scene_name})

        self.rng.shuffle(jobs)
        splits = self.cfg.get("dataset", {}).get("splits", {})
        train_ratio = float(splits.get("train", 0.8))
        val_ratio = float(splits.get("val", 0.1))
        train_cut = int(len(jobs) * train_ratio)
        val_cut = train_cut + int(len(jobs) * val_ratio)
        for idx, job in enumerate(jobs):
            if idx < train_cut:
                job["split"] = "train"
            elif idx < val_cut:
                job["split"] = "val"
            else:
                job["split"] = "test"
        return jobs

    def _generate_clip(
        self,
        clip_dir: Path,
        clip_id: str,
        split: str,
        scene_name: str,
        rng: random.Random,
    ) -> dict[str, Any]:
        self._new_stage()
        stage = self.omni_usd.get_context().get_stage()

        scene_cfg = self.cfg["scenes"][scene_name]
        lighting_profile = self._create_lighting(stage, scene_name, rng)
        terrain_profile = self._create_scene(stage, scene_name, scene_cfg, rng)
        camera_info = self._create_camera(stage)
        render_product = self.rep.create.render_product(
            CAMERA_PATH,
            (
                int(self.cfg["capture"]["width"]),
                int(self.cfg["capture"]["height"]),
            ),
        )

        annotators = self._create_annotators(render_product)
        self.app.update()

        for subdir in ("rgb", "seg", "depth", "meta", "preview"):
            ensure_dir(clip_dir / subdir)

        frame_records: list[dict[str, Any]] = []
        frames_per_clip = int(self.cfg["capture"].get("frames_per_clip", 50))
        fps = float(self.cfg["capture"].get("fps", 15))
        trajectory = self._make_trajectory(rng)

        for frame_id in range(frames_per_clip):
            timestamp = frame_id / fps
            pose = self._camera_pose_for_frame(frame_id, timestamp, trajectory, rng)
            self._set_camera_pose(stage, pose)

            self.rep.orchestrator.step(delta_time=1.0 / fps)
            frame_paths = self._write_frame_outputs(
                clip_dir=clip_dir,
                frame_id=frame_id,
                annotators=annotators,
            )

            meta = {
                "clip_id": clip_id,
                "split": split,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "scene_profile": scene_name,
                "lighting_profile": lighting_profile,
                "terrain_profile": terrain_profile,
                "camera": camera_info,
                "camera_pose": {
                    "position": pose.position,
                    "target": pose.target,
                    "up": pose.up,
                    "heading_deg": pose.heading_deg,
                    "pitch_deg": pose.pitch_deg,
                    "roll_deg": pose.roll_deg,
                },
                "paths": frame_paths,
                "label_map": self.label_map,
            }
            meta_path = clip_dir / "meta" / f"meta_{frame_id:06d}.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            frame_records.append(meta)

        preview_path = None
        if self.cfg["capture"].get("preview_video", True):
            preview_path = self._write_preview_video(clip_dir, fps)

        self._detach_annotators(annotators, render_product)
        return {
            "clip_id": clip_id,
            "split": split,
            "scene_profile": scene_name,
            "clip_dir": str(clip_dir),
            "frame_count": frames_per_clip,
            "fps": fps,
            "preview_video": preview_path,
            "lighting_profile": lighting_profile,
            "terrain_profile": terrain_profile,
            "trajectory": trajectory,
        }

    def _new_stage(self) -> None:
        self.omni_usd.get_context().new_stage()
        self.app.update()
        stage = self.omni_usd.get_context().get_stage()
        self.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.UsdGeom.SetStageUpAxis(stage, self.UsdGeom.Tokens.z)

    def _create_material(
        self,
        stage: Any,
        name: str,
        color: tuple[float, float, float],
        roughness: float = 0.85,
    ) -> Any:
        safe_name = sanitize_name(name)
        mat_path = f"/World/Materials/{safe_name}"
        material = self.UsdShade.Material.Define(stage, mat_path)
        shader = self.UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(
            self.Gf.Vec3f(*color)
        )
        shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", self.Sdf.ValueTypeNames.Float).Set(0.0)
        try:
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
        except TypeError:
            material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
        return material

    def _bind_material(self, prim: Any, material: Any) -> None:
        self.UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def _set_semantic_label(self, prim: Any, label: str) -> None:
        sem = self.Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(label)

    def _cube(
        self,
        stage: Any,
        path: str,
        translate: tuple[float, float, float],
        scale: tuple[float, float, float],
        material: Any,
        label: str,
    ) -> Any:
        cube = self.UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        xform = self.UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(self.Gf.Vec3f(*translate))
        xform.AddScaleOp().Set(self.Gf.Vec3f(*scale))
        self._bind_material(prim, material)
        self._set_semantic_label(prim, label)
        return prim

    def _cylinder(
        self,
        stage: Any,
        path: str,
        translate: tuple[float, float, float],
        radius: float,
        height: float,
        material: Any,
        label: str,
    ) -> Any:
        cylinder = self.UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        prim = cylinder.GetPrim()
        xform = self.UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(self.Gf.Vec3f(*translate))
        self._bind_material(prim, material)
        self._set_semantic_label(prim, label)
        return prim

    def _sphere(
        self,
        stage: Any,
        path: str,
        translate: tuple[float, float, float],
        scale: tuple[float, float, float],
        material: Any,
        label: str,
    ) -> Any:
        sphere = self.UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(1.0)
        prim = sphere.GetPrim()
        xform = self.UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(self.Gf.Vec3f(*translate))
        xform.AddScaleOp().Set(self.Gf.Vec3f(*scale))
        self._bind_material(prim, material)
        self._set_semantic_label(prim, label)
        return prim

    def _cone(
        self,
        stage: Any,
        path: str,
        translate: tuple[float, float, float],
        radius: float,
        height: float,
        material: Any,
        label: str,
    ) -> Any:
        cone = self.UsdGeom.Cone.Define(stage, path)
        cone.CreateRadiusAttr(radius)
        cone.CreateHeightAttr(height)
        prim = cone.GetPrim()
        xform = self.UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(self.Gf.Vec3f(*translate))
        self._bind_material(prim, material)
        self._set_semantic_label(prim, label)
        return prim

    def _create_lighting(
        self,
        stage: Any,
        scene_name: str,
        rng: random.Random,
    ) -> dict[str, float | str]:
        rand_cfg = self.cfg.get("randomization", {})
        sun_intensity = get_range(rand_cfg.get("sun_intensity"), rng, 1800)
        elevation = get_range(rand_cfg.get("sun_elevation_deg"), rng, 45)
        if "shadow" in scene_name:
            elevation = min(elevation, rng.uniform(14, 28))
            sun_intensity *= rng.uniform(0.55, 0.9)
        azimuth = get_range(rand_cfg.get("sun_azimuth_deg"), rng, 0)
        sky_intensity = get_range(rand_cfg.get("sky_intensity"), rng, 300)

        sun = self.UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(sun_intensity)
        sun.CreateAngleAttr(rng.uniform(0.2, 1.6))
        sun_xform = self.UsdGeom.Xformable(sun.GetPrim())
        sun_xform.AddRotateXYZOp().Set(
            self.Gf.Vec3f(float(90 - elevation), 0.0, float(azimuth))
        )

        sky = self.UsdLux.DomeLight.Define(stage, "/World/Sky")
        sky.CreateIntensityAttr(sky_intensity)

        return {
            "type": "low_sun_shadow" if "shadow" in scene_name else "random_sun",
            "sun_intensity": sun_intensity,
            "sun_elevation_deg": elevation,
            "sun_azimuth_deg": azimuth,
            "sky_intensity": sky_intensity,
        }

    def _create_scene(
        self,
        stage: Any,
        scene_name: str,
        scene_cfg: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        world_cfg = self.cfg.get("world", {})
        assets_cfg = self.cfg.get("assets", {})
        use_assets = bool(assets_cfg.get("use_nvidia_assets", False))
        asset_scale = float(assets_cfg.get("asset_scale", 1.0))

        width = float(world_cfg.get("terrain_width_m", 14))
        length = float(world_cfg.get("terrain_length_m", 32))
        dry_ratio = get_range(scene_cfg.get("dry_grass_ratio"), rng, 0.1)
        soil_count = get_int_range(scene_cfg.get("soil_patches"), rng, 3)
        leaf_count = get_int_range(scene_cfg.get("leaf_patches"), rng, 2)
        obstacle_count = get_int_range(scene_cfg.get("obstacles"), rng, 1)
        grass_tufts = int(world_cfg.get("grass_tufts", 350))

        # Procedural fallback materials
        mat_grass = self._create_material(stage, "grass",
            (rng.uniform(0.12, 0.24), rng.uniform(0.36, 0.58), rng.uniform(0.08, 0.18)))
        mat_dry   = self._create_material(stage, "dry_grass",
            (rng.uniform(0.45, 0.68), rng.uniform(0.37, 0.52), rng.uniform(0.15, 0.23)))
        mat_soil  = self._create_material(stage, "soil",
            (rng.uniform(0.18, 0.34), rng.uniform(0.12, 0.23), rng.uniform(0.07, 0.13)))
        mat_leaf  = self._create_material(stage, "leaf_unknown",
            (rng.uniform(0.37, 0.66), rng.uniform(0.16, 0.35), rng.uniform(0.04, 0.12)))
        mat_rock = self._create_material(stage, "rock_obstacle", (0.36, 0.35, 0.32))
        mat_wood = self._create_material(stage, "wood_obstacle", (0.34, 0.22, 0.12))
        mat_curb = self._create_material(stage, "curb",          (0.55, 0.55, 0.5))
        mat_road = self._create_material(stage, "road",          (0.08, 0.09, 0.09))

        # ── Ground / terrain ─────────────────────────────────────────────
        terrain_loaded = False
        if use_assets:
            ground_url = assets_cfg.get("terrain_ground", "")
            if ground_url:
                prim = self._load_asset(
                    stage, "/World/Terrain/GrassGround", ground_url,
                    translate=(0.0, 0.0, 0.0), scale=asset_scale, label="grass",
                )
                if prim is not None:
                    terrain_loaded = True
                    print("[Assets] Terrain ground loaded from Nucleus.")
        if not terrain_loaded:
            self._cube(stage, "/World/Terrain/GrassBase",
                       (0.0, 0.0, -0.03), (width, length, 0.06), mat_grass, "grass")

        # ── Soil patches (always procedural) ─────────────────────────────
        for idx in range(soil_count):
            sx = rng.uniform(0.6, 2.4)
            sy = rng.uniform(0.8, 3.2)
            x  = rng.uniform(-width  * 0.42, width  * 0.42)
            y  = rng.uniform(-length * 0.42, length * 0.42)
            self._cube(stage, f"/World/Terrain/SoilPatch_{idx:03d}",
                       (x, y, 0.005), (sx, sy, 0.012), mat_soil, "soil")

        leaf_urls = assets_cfg.get("leaves", [])
        for idx in range(leaf_count):
            x = rng.uniform(-width  * 0.44, width  * 0.44)
            y = rng.uniform(-length * 0.44, length * 0.44)
            placed = False
            if use_assets and leaf_urls:
                chosen_url = rng.choice(leaf_urls)
                leaf_scale = rng.uniform(0.8, 1.5) * asset_scale
                rot_z = rng.uniform(0.0, 360.0)
                prim = self._load_asset(
                    stage,
                    f"/World/Terrain/LeafPatch_{idx:03d}",
                    chosen_url,
                    translate=(x, y, 0.0),
                    rotate_xyz=(0.0, 0.0, rot_z),
                    scale=leaf_scale,
                    label="unknown",
                )
                if prim is not None:
                    placed = True
                    
            if not placed:
                radius = rng.uniform(0.25, 0.9)
                height = rng.uniform(0.006, 0.016)
                self._cylinder(stage, f"/World/Terrain/LeafPatch_{idx:03d}",
                               (x, y, height * 0.5 + 0.015), radius, height, mat_leaf, "unknown")

        # ── Grass tufts ───────────────────────────────────────────────────
        grass_urls = assets_cfg.get("grass", [])
        dry_urls   = assets_cfg.get("dry_grass", [])

        for idx in range(grass_tufts):
            is_dry = rng.random() < dry_ratio
            x = rng.uniform(-width  * 0.48, width  * 0.48)
            y = rng.uniform(-length * 0.48, length * 0.48)
            placed = False
            if use_assets:
                url_list = dry_urls if (is_dry and dry_urls) else grass_urls
                if url_list:
                    prim = self._load_asset(
                        stage, f"/World/GrassTufts/Tuft_{idx:04d}",
                        rng.choice(url_list),
                        translate=(x, y, 0.0),
                        rotate_xyz=(0.0, 0.0, rng.uniform(0.0, 360.0)),
                        scale=rng.uniform(0.8, 1.4) * asset_scale,
                        label="dry_grass" if is_dry else "grass",
                    )
                    if prim is not None:
                        placed = True
            if not placed:
                mat    = mat_dry if is_dry else mat_grass
                radius = rng.uniform(0.012, 0.035)
                height = rng.uniform(0.035, 0.15)
                self._cone(stage, f"/World/GrassTufts/Tuft_{idx:04d}",
                           (x, y, height * 0.5), radius, height, mat, "grass")

        # ── Obstacles ─────────────────────────────────────────────────────
        rock_urls = assets_cfg.get("rocks", [])

        for idx in range(obstacle_count):
            x = rng.uniform(-width  * 0.42, width  * 0.42)
            y = rng.uniform(-length * 0.34, length * 0.44)
            placed = False
            if use_assets and rock_urls:
                prim = self._load_asset(
                    stage, f"/World/Obstacles/Rock_{idx:03d}",
                    rng.choice(rock_urls),
                    translate=(x, y, 0.0),
                    rotate_xyz=(0.0, 0.0, rng.uniform(0.0, 360.0)),
                    scale=rng.uniform(0.3, 1.0) * asset_scale,
                    label="obstacle",
                )
                if prim is not None:
                    placed = True
            if not placed:
                if rng.random() < 0.5:
                    sc = (rng.uniform(0.18, 0.45), rng.uniform(0.16, 0.38), rng.uniform(0.08, 0.28))
                    self._sphere(stage, f"/World/Obstacles/Rock_{idx:03d}",
                                 (x, y, sc[2]), sc, mat_rock, "obstacle")
                else:
                    h = rng.uniform(0.35, 1.0)
                    r = rng.uniform(0.06, 0.18)
                    self._cylinder(stage, f"/World/Obstacles/Post_{idx:03d}",
                                   (x, y, h * 0.5), r, h, mat_wood, "obstacle")

        # ── Curb / road / shadow (always procedural) ──────────────────────
        if "obstacle" in scene_name or "shadow" in scene_name:
            self._cube(stage, "/World/Boundary/CurbLeft",
                       (-width * 0.48, 1.0, 0.055), (0.16, length * 0.8, 0.11),
                       mat_curb, "curb")
            self._cube(stage, "/World/Boundary/RoadRight",
                       (width * 0.48, 1.0, 0.01), (0.9, length * 0.8, 0.02),
                       mat_road, "road")
            self._cube(stage, "/World/ShadowCaster",
                       (-width * 0.2, -length * 0.1, 2.1), (0.18, length * 0.5, 1.8),
                       mat_wood, "obstacle")

        return {
            "scene_name":        scene_name,
            "terrain_width_m":   width,
            "terrain_length_m":  length,
            "soil_patches":      soil_count,
            "leaf_patches":      leaf_count,
            "obstacles":         obstacle_count,
            "grass_tufts":       grass_tufts,
            "dry_grass_ratio":   dry_ratio,
            "use_nvidia_assets": use_assets,
        }




    def _create_camera(self, stage: Any) -> dict[str, Any]:
        camera_cfg = self.cfg["camera"]
        camera = self.UsdGeom.Camera.Define(stage, CAMERA_PATH)
        camera.CreateClippingRangeAttr(self.Gf.Vec2f(0.02, 80.0))
        aperture = 20.955
        fov = float(camera_cfg.get("horizontal_fov_deg", 90))
        focal_length = aperture / (2.0 * math.tan(math.radians(fov) / 2.0))
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateFocalLengthAttr(focal_length)

        xform = self.UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTransformOp()

        return {
            "height_m": float(camera_cfg.get("height_m", 0.5)),
            "pitch_deg": float(camera_cfg.get("pitch_deg", 32)),
            "horizontal_fov_deg": fov,
            "focal_length": focal_length,
            "resolution": [
                int(self.cfg["capture"]["width"]),
                int(self.cfg["capture"]["height"]),
            ],
        }

    def _make_trajectory(self, rng: random.Random) -> dict[str, float]:
        camera_cfg = self.cfg["camera"]
        speed = get_range(camera_cfg.get("forward_speed_mps"), rng, 0.45)
        turn_rate = get_range(camera_cfg.get("turn_rate_degps"), rng, 0.0)
        world_cfg = self.cfg.get("world", {})
        width = float(world_cfg.get("terrain_width_m", 14))
        length = float(world_cfg.get("terrain_length_m", 32))
        return {
            "start_x": rng.uniform(-width * 0.18, width * 0.18),
            "start_y": -length * 0.42,
            "start_heading_deg": rng.uniform(-5, 5),
            "forward_speed_mps": speed,
            "turn_rate_degps": turn_rate,
        }

    def _camera_pose_for_frame(
        self,
        frame_id: int,
        timestamp: float,
        trajectory: dict[str, float],
        rng: random.Random,
    ) -> CameraPose:
        camera_cfg = self.cfg["camera"]
        jitter_cfg = camera_cfg.get("jitter", {})
        height = float(camera_cfg.get("height_m", 0.5))
        base_pitch = float(camera_cfg.get("pitch_deg", 32))

        position_noise = float(jitter_cfg.get("position_m", 0.0))
        yaw_noise = float(jitter_cfg.get("yaw_deg", 0.0))
        pitch_noise = float(jitter_cfg.get("pitch_deg", 0.0))
        roll_noise = float(jitter_cfg.get("roll_deg", 0.0))

        heading_deg = (
            trajectory["start_heading_deg"]
            + trajectory["turn_rate_degps"] * timestamp
            + rng.uniform(-yaw_noise, yaw_noise)
        )
        pitch_deg = clamp(
            base_pitch + rng.uniform(-pitch_noise, pitch_noise),
            10.0,
            70.0,
        )
        roll_deg = rng.uniform(-roll_noise, roll_noise)

        heading_rad = math.radians(heading_deg)
        direction = (math.sin(heading_rad), math.cos(heading_rad), 0.0)
        x = (
            trajectory["start_x"]
            + math.sin(math.radians(trajectory["turn_rate_degps"]) * timestamp)
            * 0.7
            + rng.uniform(-position_noise, position_noise)
        )
        y = (
            trajectory["start_y"]
            + trajectory["forward_speed_mps"] * timestamp
            + rng.uniform(-position_noise, position_noise)
        )
        z = height + rng.uniform(-position_noise, position_noise)

        lookahead = z / math.tan(math.radians(pitch_deg))
        target = (
            x + direction[0] * lookahead,
            y + direction[1] * lookahead,
            0.0,
        )
        forward = self._normalize(
            (target[0] - x, target[1] - y, target[2] - z)
        )
        up = self._roll_up_vector((0.0, 0.0, 1.0), forward, math.radians(roll_deg))
        return CameraPose(
            position=(x, y, z),
            target=target,
            up=up,
            heading_deg=heading_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )

    def _set_camera_pose(self, stage: Any, pose: CameraPose) -> None:
        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        xform = self.UsdGeom.Xformable(camera_prim)
        ops = xform.GetOrderedXformOps()
        if not ops:
            op = xform.AddTransformOp()
        else:
            op = ops[0]
        matrix = self.Gf.Matrix4d().SetLookAt(
            self.Gf.Vec3d(*pose.position),
            self.Gf.Vec3d(*pose.target),
            self.Gf.Vec3d(*pose.up),
        ).GetInverse()
        op.Set(matrix)

    def _create_annotators(self, render_product: Any) -> dict[str, Any]:
        annotators: dict[str, Any] = {}
        capture_cfg = self.cfg["capture"]

        if capture_cfg.get("rgb", True):
            annotators["rgb"] = self.rep.AnnotatorRegistry.get_annotator("rgb")
            annotators["rgb"].attach([render_product])

        if capture_cfg.get("semantic", True):
            annotators["semantic"] = self.rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation",
                init_params={"colorize": False},
            )
            annotators["semantic"].attach([render_product])

        if capture_cfg.get("depth", True):
            annotators["depth"] = self.rep.AnnotatorRegistry.get_annotator(
                "distance_to_camera"
            )
            annotators["depth"].attach([render_product])

        return annotators

    def _write_frame_outputs(
        self,
        clip_dir: Path,
        frame_id: int,
        annotators: dict[str, Any],
    ) -> dict[str, str]:
        paths: dict[str, str] = {}

        if "rgb" in annotators:
            rgb_data = annotators["rgb"].get_data()
            rgb_path = clip_dir / "rgb" / f"rgb_{frame_id:06d}.png"
            self._write_rgb_png(rgb_path, rgb_data)
            paths["rgb"] = str(rgb_path)

        if "semantic" in annotators:
            packet = annotators["semantic"].get_data()
            sem_path = clip_dir / "seg" / f"seg_{frame_id:06d}.png"
            mapping_path = clip_dir / "seg" / f"seg_mapping_{frame_id:06d}.json"
            self._write_semantic_png(sem_path, mapping_path, packet)
            paths["semantic"] = str(sem_path)
            paths["semantic_mapping"] = str(mapping_path)

        if "depth" in annotators:
            depth_data = annotators["depth"].get_data()
            depth_path = self._write_depth(clip_dir / "depth", frame_id, depth_data)
            paths["depth"] = str(depth_path)

        return paths

    def _write_rgb_png(self, path: Path, data: Any) -> None:
        arr = self.np.asarray(data)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = self.np.clip(arr, 0, 255).astype(self.np.uint8)
        self.Image.fromarray(arr).save(path)

    def _write_semantic_png(self, path: Path, mapping_path: Path, packet: Any) -> None:
        if isinstance(packet, dict):
            raw = self.np.asarray(packet.get("data"))
            info = packet.get("info", {})
        else:
            raw = self.np.asarray(packet)
            info = {}

        if raw.ndim == 3:
            raw = raw[..., 0]

        semantic_lookup = self._semantic_lookup(info)
        mapped = self.np.zeros(raw.shape, dtype=self.np.uint8)
        if semantic_lookup:
            for raw_id, target_id in semantic_lookup.items():
                mapped[raw == raw_id] = target_id
        else:
            mapped = self.np.clip(raw, 0, 255).astype(self.np.uint8)

        self.Image.fromarray(mapped, mode="L").save(path)
        mapping_path.write_text(
            json.dumps(
                {
                    "label_map": self.label_map,
                    "replicator_info": self._json_safe(info),
                    "raw_to_target": semantic_lookup,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_depth(self, depth_dir: Path, frame_id: int, data: Any) -> Path:
        arr = self.np.asarray(data).astype(self.np.float32)
        exr_path = depth_dir / f"depth_{frame_id:06d}.exr"
        try:
            import imageio.v3 as iio  # type: ignore

            iio.imwrite(exr_path, arr)
            return exr_path
        except Exception:
            npy_path = depth_dir / f"depth_{frame_id:06d}.npy"
            self.np.save(npy_path, arr)
            return npy_path

    def _semantic_lookup(self, info: dict[str, Any]) -> dict[int, int]:
        id_to_labels = (
            info.get("idToLabels")
            or info.get("idToSemantics")
            or info.get("semanticIdToLabels")
            or {}
        )
        lookup: dict[int, int] = {}
        if not isinstance(id_to_labels, dict):
            return lookup

        for raw_id, labels in id_to_labels.items():
            try:
                raw_int = int(raw_id)
            except (TypeError, ValueError):
                continue
            label = self._extract_label(labels)
            normalized = self._normalize_label(label)
            lookup[raw_int] = self.label_map.get(
                normalized,
                self.label_map.get("background", 0),
            )
        lookup.setdefault(0, self.label_map.get("background", 0))
        return lookup

    def _extract_label(self, labels: Any) -> str:
        if isinstance(labels, str):
            return labels
        if isinstance(labels, dict):
            for key in ("class", "Class", "semanticLabel", "label", "data"):
                value = labels.get(key)
                if isinstance(value, str):
                    return value
            for value in labels.values():
                nested = self._extract_label(value)
                if nested:
                    return nested
        if isinstance(labels, (list, tuple)):
            for value in labels:
                nested = self._extract_label(value)
                if nested:
                    return nested
        return ""

    def _normalize_label(self, label: str) -> str:
        lowered = label.lower()
        for known in self.label_map.keys():
            if lowered == known or known in lowered:
                return known
        if "dirt" in lowered or "mud" in lowered:
            return "soil"
        if "leaf" in lowered or "leaves" in lowered:
            return "unknown"
        if "stone" in lowered or "rock" in lowered or "post" in lowered:
            return "obstacle"
        return "background"

    def _write_preview_video(self, clip_dir: Path, fps: float) -> str | None:
        if not self.cfg["capture"].get("preview_video", True):
            return None
        rgb_files = sorted((clip_dir / "rgb").glob("rgb_*.png"))
        if not rgb_files:
            return None
        try:
            import cv2  # type: ignore
        except ImportError:
            print("OpenCV not available; skipping MP4 preview.")
            return None

        first = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
        if first is None:
            return None
        height, width = first.shape[:2]
        preview_path = clip_dir / "preview" / "rgb_preview.mp4"
        writer = cv2.VideoWriter(
            str(preview_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        for file_path in rgb_files:
            frame = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if frame is not None:
                writer.write(frame)
        writer.release()
        return str(preview_path)

    def _detach_annotators(self, annotators: dict[str, Any], render_product: Any) -> None:
        for annotator in annotators.values():
            try:
                annotator.detach([render_product])
            except Exception:
                try:
                    annotator.detach()
                except Exception:
                    pass

    def _normalize(self, vector: tuple[float, float, float]) -> tuple[float, float, float]:
        mag = math.sqrt(sum(component * component for component in vector))
        if mag <= 1e-9:
            return (0.0, 1.0, 0.0)
        return tuple(component / mag for component in vector)  # type: ignore[return-value]

    def _roll_up_vector(
        self,
        up: tuple[float, float, float],
        axis: tuple[float, float, float],
        angle_rad: float,
    ) -> tuple[float, float, float]:
        ux, uy, uz = self._normalize(axis)
        vx, vy, vz = up
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        dot = ux * vx + uy * vy + uz * vz
        cross = (
            uy * vz - uz * vy,
            uz * vx - ux * vz,
            ux * vy - uy * vx,
        )
        rolled = (
            vx * cos_a + cross[0] * sin_a + ux * dot * (1.0 - cos_a),
            vy * cos_a + cross[1] * sin_a + uy * dot * (1.0 - cos_a),
            vz * cos_a + cross[2] * sin_a + uz * dot * (1.0 - cos_a),
        )
        return self._normalize(rolled)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_dir:
        deep_update(cfg, "dataset.output_dir", args.output_dir)
    if args.clips_per_scene is not None:
        deep_update(cfg, "capture.clips_per_scene", args.clips_per_scene)
    if args.frames_per_clip is not None:
        deep_update(cfg, "capture.frames_per_clip", args.frames_per_clip)
    if args.headless:
        deep_update(cfg, "simulation.headless", True)
    if args.visible:
        deep_update(cfg, "simulation.headless", False)
    if args.no_preview:
        deep_update(cfg, "capture.preview_video", False)


def main() -> int:
    args = parse_args()
    cfg = load_mapping_file(args.config)
    label_map = load_mapping_file(args.label_map)
    apply_cli_overrides(cfg, args)

    sim_cfg = cfg.get("simulation", {})
    launch_config = {
        "headless": bool(sim_cfg.get("headless", True)),
        "renderer": str(sim_cfg.get("renderer", "RayTracedLighting")),
        "samples_per_pixel_per_frame": int(sim_cfg.get("samples_per_pixel", 32)),
        "width": int(cfg["capture"].get("width", 960)),
        "height": int(cfg["capture"].get("height", 540)),
    }

    try:
        from omni.isaac.kit import SimulationApp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "This script must be run with Isaac Sim's Python environment."
        ) from exc

    simulation_app = SimulationApp(launch_config)
    try:
        generator = IsaacGrassDatasetGenerator(simulation_app, cfg, label_map)
        generator.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
