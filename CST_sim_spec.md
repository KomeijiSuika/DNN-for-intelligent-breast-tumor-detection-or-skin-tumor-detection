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

## 仿真设置（波源与探测面）

- 求解器：Frequency Domain（或 Time Domain，根据偏好）
- 频点：30 GHz（或 29.5–30.5 GHz 小扫频）
- 激励：
  - 使用 Plane Wave 激励，从 metasurface 入射侧沿 ±z 方向正入射（normal incidence）。
  - 建议让平面波从负 z 指向正 z，元表面 3 层位于 z≈0 之后，入射波先经过空气再入射到第一层。
- 探测面：
  - 在第三层之后的自由空间中放置一个平面场监视器（E-field monitor），其 z 位置对应脚本中最后一个传播距离 `z_list_m[3]` 之后的探测平面。
  - 在该平面上记录 |E|² 强度分布，用于和 Python 仿真 `simulate.py` 的 `intensity` 做对比。

## 边界与模型几何

- 周期单元模式：x/y 周期边界，z 方向开边或 PML。
- 有限面阵：在 z 双向使用 PML，仿真区域上下至少留 1 λ 空间。

## 输出建议

- S 参数（S11, S21）导出为 `.snp` 或 `.csv`
- 透射/反射相位与幅度表格（便于与训练结果比较）
- 近场（E/H）平面快照与探测面强度（CSV 或图像）
- 若用于制造：为每个像素导出几何参数表（CSV），并生成 CAD 布局（STEP/IGES）

## EMNIST 字母的汇聚区域与几何关系

### 几何尺寸与距离（对应 `configs/emnist_letters.yaml`）

- 像素间距：`pixel_size_m = 1.0e-3` → 每个像素边长 1 mm。
- metasurface 横向尺寸：128×128 像素 → 128 mm × 128 mm 的正方形孔径（下采样到 64×64、32×32 时，物理尺寸仍保持约 128 mm，仅像素变粗）。
- 轴向距离（忽略层厚，取中心位置，单位 m）：
  - 输入平面（字母波前）记为 z = 0；
  - 第 1 层中心：z ≈ 0.10；
  - 第 2 层中心：z ≈ 0.20；
  - 第 3 层中心：z ≈ 0.30；
  - 探测平面（接收面 / detector plane）：z ≈ 0.40（即第 3 层后方约 0.10 m）。
- 在 CST 中：
  - 平面波从 −z 方向入射到三层结构（normal incidence），覆盖至少 128 mm × 128 mm 区域；
  - 在 z ≈ 0.40 m 处放置 E-field monitor，记录 |E|²，作为聚焦/分类的探测面。

### 2×13 汇聚区域（detector regions）

- 探测面被离散为 128×128 像素的正方形平面（1 mm 像素间距）。
- 分类器在该平面上布置一个 2×13 的小方块阵列作为“汇聚区域”（detector regions）：
  - 配置参数：`rows: 2`, `cols: 13`, `square: 7`, `gap: 2`, `margin: 6`；
  - 含义：从左上角留出 6 像素边界后，以 7×7 像素的小方块、相邻方块之间间隔 2 像素，铺出 2 行 13 列的网格。
- 字母到方块的对应关系：
  - 上排 13 个方块（从左到右）：字母 A–M；
  - 下排 13 个方块（从左到右）：字母 N–Z。
- 训练目标：不同字母在探测面上的强度分布，使得对应方块内的积分能量 ∫|E|² dA 最大，相当于“焦点”落在该格子里。
- 在 CST 后处理中，可按上述 2×13 网格划分探测平面，对每个方块积分 |E|²：
  - 给定字母入射下，找到能量最大的方块，即该字母的汇聚点；
  - 将其与 Python 侧 `simulate.py` / `predict.py` 给出的类别进行对比验证。

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
