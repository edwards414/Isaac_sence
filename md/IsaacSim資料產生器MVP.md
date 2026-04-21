# Isaac Sim 草地資料產生器 MVP

這一版的目標是先把 pipeline 跑通：

- 4 種 procedural 場景：正常草地、草土混合、枯草落葉、陰影障礙物
- 固定車載相機視角
- 相機沿前進路徑產生連續 clip
- 每幀輸出 RGB、semantic mask、depth、metadata
- 每段 clip 額外輸出 RGB preview mp4

訓練時建議讀逐幀資料，不要直接讀 mp4。mp4 主要拿來人工檢查畫面是否合理。

## 檔案

```text
configs/
  dataset_config.yaml
  label_map.yaml

scripts/
  generate_grass_dataset.py
```

## 快速測試

請用 Isaac Sim 內建 Python 執行，不要用系統 Python。

Windows 範例：

```powershell
cd d:\project\isaac_sence
<ISAAC_SIM_ROOT>\python.bat scripts\generate_grass_dataset.py `
  --config configs\dataset_config.yaml `
  --label-map configs\label_map.yaml `
  --clips-per-scene 1 `
  --frames-per-clip 10 `
  --visible
```

如果要 headless 跑：

```powershell
cd d:\project\isaac_sence
<ISAAC_SIM_ROOT>\python.bat scripts\generate_grass_dataset.py `
  --config configs\dataset_config.yaml `
  --label-map configs\label_map.yaml `
  --clips-per-scene 1 `
  --frames-per-clip 10 `
  --headless
```

確認小測試能跑完後，再跑完整 MVP：

```powershell
cd d:\project\isaac_sence
<ISAAC_SIM_ROOT>\python.bat scripts\generate_grass_dataset.py `
  --config configs\dataset_config.yaml `
  --label-map configs\label_map.yaml `
  --headless
```

## 輸出格式

預設輸出到：

```text
dataset/
  train/
    clip_000001/
      rgb/
        rgb_000000.png
      seg/
        seg_000000.png
        seg_mapping_000000.json
      depth/
        depth_000000.exr 或 depth_000000.npy
      meta/
        meta_000000.json
      preview/
        rgb_preview.mp4
  val/
  test/
  dataset_manifest.json
```

semantic mask 會被轉成固定 class id：

```text
0 background
1 grass
2 soil
3 obstacle
4 unknown
5 curb
6 road
```

如果 EXR writer 在目前 Isaac Sim Python 環境不可用，depth 會自動 fallback 成 `.npy`。

## 目前 MVP 的限制

這版先追求可產生與可訓練，不追求視覺擬真極限。

- 草地是 procedural 幾何與材質，不是高品質 vegetation asset
- 葉子目前歸到 `unknown`
- depth 優先輸出 EXR，失敗時輸出 NPY
- preview mp4 需要 Isaac Sim Python 環境有 OpenCV，否則會略過
- 真實相機特性像曝光、白平衡、motion blur 之後可以再補

## 下一步建議

第一階段先跑：

```text
4 scenes x 1 clip x 10 frames
```

確認：

- Isaac Sim 可以順利啟動
- RGB 有畫面
- seg mask class id 正確
- metadata 中 camera pose 正確
- preview mp4 可打開

第二階段再跑：

```text
4 scenes x 20 clips x 50 frames = 4000 frames
```

第三階段再加入：

- 真實草地 USD / MDL 材質
- 相機 noise、gamma、曝光、白平衡
- motion blur
- 更貼近真車的 camera intrinsic
- 與真實資料混合訓練的 dataset loader
