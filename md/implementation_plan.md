# 使用 NVIDIA Assets + GPU RTX 渲染升級計畫

## 目標
將場景從程式生成的幾何形狀（Cube/Cylinder/Cone）升級為使用 **NVIDIA Omniverse Assets**（草地、植被、地形）搭配 **RTX GPU 渲染**（PathTracing 或 RayTracedLighting）。

---

## 目前架構 vs 目標架構

| 項目 | 現在 | 目標 |
|---|---|---|
| 草地 | `UsdGeom.Cone`（程式化幾何） | NVIDIA Assets USD 草地模型 |
| 地形 | `UsdGeom.Cube` 壓扁的平面 | Assets 地形 USD |
| 障礙物 | `UsdGeom.Sphere/Cylinder` | Assets 岩石/木樁 USD |
| 渲染器 | `RayTracedLighting` 32spp | `PathTracing`（高品質）或保留 RTX |
| 光源 | DistantLight + DomeLight（程式化） | 可選加入 HDRI DomeLight |

---

## 需要確認的問題

> [!IMPORTANT]
> **Q1：你的 Nucleus 伺服器狀態**
> - 是否有安裝並運行本地 Nucleus？（`omniverse://localhost/`）
> - 或使用 NVIDIA 雲端 Assets CDN？（HTTP URL）
> - 還是已下載 Assets 到本機資料夾？

> [!IMPORTANT]
> **Q2：草地 Asset 來源**
> 請先開啟 Isaac Sim UI → Window → Browsers → NVIDIA Assets，搜尋 `grass` 或 `vegetation`，找到想使用的 USD 路徑（例如 `omniverse://localhost/NVIDIA/Assets/ArchVis/Vegetation/Plants/...`），提供路徑讓我設定。

> [!IMPORTANT]
> **Q3：渲染品質 vs 速度**
> - `RayTracedLighting`（快，~30 spp）→ 適合大量資料生成
> - `PathTracing`（高品質但慢 5-10x）→ 適合少量高品質樣本
> - 你的 GPU 型號？

---

## 提議的改動

### 1. `configs/dataset_config.yaml` [MODIFY]

新增 `assets` 區塊：
```yaml
assets:
  use_nvidia_assets: true
  nucleus_root: ""   # 留空則自動用 get_assets_root_path()
  grass:
    - "NVIDIA/Assets/ArchVis/Vegetation/Plants/GrassPatches/grass_01.usd"
    - "NVIDIA/Assets/ArchVis/Vegetation/Plants/GrassPatches/grass_02.usd"
  dry_grass:
    - "NVIDIA/Assets/ArchVis/Vegetation/Plants/GrassPatches/dry_grass_01.usd"
  rocks:
    - "NVIDIA/Assets/ArchVis/Exterior/Props/Rocks/rock_01.usd"
  terrain_ground:
    "NVIDIA/Assets/ArchVis/Exterior/Surfaces/Grass/grass_ground.usd"

simulation:
  renderer: PathTracing     # 改為 GPU PathTracing
  samples_per_pixel: 64     # 提高品質
```

---

### 2. `scripts/generate_grass_dataset.py` [MODIFY]

#### 新增 `_load_asset()` 方法
```python
def _load_asset(self, stage, prim_path, asset_url, 
                translate=(0,0,0), rotate=(0,0,0), scale=(1,1,1),
                label="", instanceable=True):
    import omni.kit.commands
    omni.kit.commands.execute(
        'CreateReferenceCommand',
        usd_context=self.omni_usd.get_context(),
        path_to=prim_path,
        asset_path=asset_url,
        instanceable=instanceable,
    )
    prim = stage.GetPrimAtPath(prim_path)
    xform = self.UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(self.Gf.Vec3f(*translate))
    xform.AddRotateXYZOp().Set(self.Gf.Vec3f(*rotate))
    xform.AddScaleOp().Set(self.Gf.Vec3f(*scale))
    if label:
        self._set_semantic_label(prim, label)
    return prim
```

#### 修改 `_create_scene()` 方法
- 根據 `cfg["assets"]["use_nvidia_assets"]` 開關決定使用 Assets 還是舊的幾何形狀
- 保留向後兼容：Assets 找不到時 fallback 到幾何形狀

#### 新增 `_get_nucleus_root()` 方法
```python
def _get_nucleus_root(self):
    root = self.cfg.get("assets", {}).get("nucleus_root", "")
    if not root:
        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path
            root = get_assets_root_path()
        except ImportError:
            root = "omniverse://localhost"
    return root
```

---

## 實作順序

1. 修改 `dataset_config.yaml`，加入 `assets` 區塊
2. 在 `IsaacGrassDatasetGenerator.__init__()` 中加入 `omni.kit.commands` import
3. 新增 `_get_nucleus_root()` 和 `_load_asset()` helper
4. 修改 `_create_scene()` 加入 assets 分支（保留幾何 fallback）
5. 更新渲染器設定（PathTracing/RTX）

---

## 驗證計畫

```bash
make test-smoke-visual   # 開啟 UI，確認場景有 Assets 載入
```

確認：
- NVIDIA 草地模型出現在 Viewport
- seg mask 正確標記為 `grass`
- GPU 使用率明顯（nvidia-smi）
- 輸出 PNG 明顯比幾何形狀更真實

---

## 開放問題

1. 請確認 Nucleus 是否運行、或提供 Assets 路徑
2. 確認 GPU 型號（影響 PathTracing spp 設定）
3. 確認是否要保留幾何形狀 fallback（建議保留）
