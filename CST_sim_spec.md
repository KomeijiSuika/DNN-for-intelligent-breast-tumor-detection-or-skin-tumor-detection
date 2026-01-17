# CST Simulation Specification

## 项目 / 目标
在 CST 中对训练得到的 metasurface 进行 30 GHz（λ ≈ 10 mm）工作点的电磁仿真，验证透射/反射及探测面强度，并导出可用于制造的几何参数。

## 关键物理参数
- 工作频率：30 GHz
- 真空波长：λ0 = 10 mm

## 材料（Vero White Plus）
- 相对介电常数：εr = 2.802
- 损耗正切：tanδ = 0.0357
- 计算的等效电导率（用于需要 σ 的仿真项）：
  - ω = 2π·30e9 rad/s
  - ε0 = 8.854e-12 F/m
  - σ ≈ tanδ · ω · ε0 · εr ≈ 0.167 S/m

> 注：若 CST 要求以复介电常数输入，可用 ε = εr - j·(σ / (ω·ε0))，或直接使用 `tanδ` 与 `εr`（CST 常能接受）。

## 网格与单元
- 单像素（unit cell）尺寸：1.0 mm（根据 `configs/emnist_letters.yaml` 中 `pixel_size_m: 1.0e-3`）
- 网格建议：介质中每波长 ≥ 10–20 个单元（建议 λ/20 → 0.5 mm），关键小特征处 ≤ 0.2–0.3 mm。
- 若使用周期单元仿真（unit cell）：使用周期边界条件；若模拟有限面阵使用 PML/开边界。

## 仿真设置
- 求解器：Frequency Domain（或 Time Domain，根据偏好）
- 频点：30 GHz（或 29.5–30.5 GHz 小扫频）
- 激励：平面波（normal incidence，极化按设计—例如 x 极化）

## 边界与模型几何
- 周期单元模式：x/y 周期边界，z 方向开边或 PML
- 有限面阵：在 z 双向使用 PML，仿真区域上下至少留 1 λ 空间

## 输出建议
- S 参数（S11, S21）导出为 `.snp` 或 `.csv`
- 透射/反射相位与幅度表格（便于与训练结果比较）
- 近场（E/H）平面快照与探测面强度（CSV 或图像）
- 若用于制造：为每个像素导出几何参数表（CSV），并生成 CAD 布局（STEP/IGES）

## 文件与工作流
1. 训练得到 `phase_masks.npz`（包含 `layer1_phase_rad`, `layer2_phase_rad`, `layer3_phase_rad`）
2. 使用 LUT（CSV）进行相位到几何的量化（nearest neighbor），得到每层的 `geom_param` 数组和 LUT 索引
3. 将 `geom_param` 转为 CST 单元几何（单位保持为 mm），生成 CAD 布局并导入 CST
4. 在 CST 中设置材料、网格、激励与求解器并运行仿真
5. 导出 S 参数与场数据，用于与 `simulate.py` 输出或实验数据对比

## 常用文件位置（项目内）
- 配置示例：[configs/default.yaml](configs/default.yaml)
- EMNIST 特定配置：[configs/emnist_letters.yaml](configs/emnist_letters.yaml)
- 示例 LUT：[assets/meta_atom_lut_example.csv](assets/meta_atom_lut_example.csv)
- 导出脚本（将生成的几何参数用于 CST）：`scripts/generate_cst_params.py`

## 快速命令（示例）
```bash
# 生成 CST 参数（如果已有 phase_masks.npz 与 LUT）：
python scripts/generate_cst_params.py --config configs/emnist_letters.yaml \
    --phase_npz outputs/.../phase_masks.npz \
    --lut_csv assets/meta_atom_lut_example.csv \
    --out_dir outputs/cst_params

# 只生成规格与材料参数（若没有 phase_npz）：
python scripts/generate_cst_params.py --config configs/emnist_letters.yaml --out_dir outputs/cst_params
```

---

如果你需要，我可以把 `outputs/cst_params` 中的 `geom_param` 再转换为 STEP/IGES 布局脚本（需知道元胞几何模版），或者帮你把当前 `assets/meta_atom_lut_example.csv` 转为更高分辨率的 LUT。