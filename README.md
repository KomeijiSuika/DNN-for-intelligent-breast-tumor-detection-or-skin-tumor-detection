# Metasurface-based Diffractive Neural Network (3-layer) | EMNIST Letters (Microwave, Simulation)

本课设分两步：

1) **算法与仿真**：设计一个**微波频段**的三层衍射神经网络（phase-only metasurface），完成 **EMNIST 字母分类（26 类）**。
2) **工程验证**：用 3D 打印材料实现相位层（或等效结构），并在 CST 中选取若干案例验证。

当前仓库主要完成第 1 步（算法与仿真），并提供导出结果（相位掩膜、厚度图、预测 CSV）以对接第 2 步。

你需要设计 3 层超表面的 meta-atom 分布（等效为每层的空间相位分布 $\phi(x,y)\in[0,2\pi)$ ），让入射电磁波在自由空间传播 + 相位调制后，在探测面形成**26 个类别对应的能量分布**，从而完成分类。

本实现参考了仓库中已有的衍射网络资料（如 [onn-simulation](onn-simulation/README.md) 的“传播层 + 调制层 + 探测层”思想），但这里**严格按题目要求：3 层、相位-only、仿真为主**。

## 1. 物理原理（电磁/衍射到可训练网络）

### 1.1 标量衍射与角谱法（Angular Spectrum）
在均匀介质中，单频电磁场可表示为复振幅形式：
$$E(x,y,z,t)=\Re\{U(x,y,z) e^{j\omega t}\}.$$

在近似为标量波（忽略极化耦合、只考虑一个分量）时，平面 $z$ 上的场 $U(x,y,z)$ 的传播可通过角谱法表示：

1) 对 $z=0$ 平面的场做二维傅里叶变换得到角谱 $A(f_x,f_y)$。

2) 每个空间频率分量在传播距离 $z$ 后乘上传输函数：
$$H(f_x,f_y;z)=\exp\left(j k z\sqrt{1-(\lambda f_x)^2-(\lambda f_y)^2}\right),$$
其中 $k=\frac{2\pi n_0}{\lambda}$，$n_0$ 为背景折射率。

3) 逆傅里叶变换得到传播后的场。

在实现中，为避免非传播(消逝)分量带来的数值问题，默认会对 $1-(\lambda f_x)^2-(\lambda f_y)^2<0$ 的部分做 band-limit mask（只保留传播波）。

### 1.2 超表面等效为“相位掩膜”
题目给定 meta-atom 结构可实现 $0\sim2\pi$ 连续相位调制。对标量场近似下，一层超表面可等效为：
$$U_\text{out}(x,y)=U_\text{in}(x,y)\,e^{j\phi(x,y)},\quad \phi\in[0,2\pi).$$

因此 3 层超表面 + 4 段自由空间传播（输入→L1→L2→L3→探测面）可以写成“可微物理算子”的串联。

### 1.3 为什么它是“神经网络”
网络的“参数”就是每层的相位分布 $\phi_1,\phi_2,\phi_3$；
前向传播就是电磁波的传播与调制；
损失函数把探测面的光强分布映射为分类分数（例如两个探测区域能量之差），用梯度下降即可训练相位掩膜。

本仓库实现一个多分类探测器：在探测面选取多个窗口区域 $R_c$（默认 2×13 网格，共 26 个），以
$$s_c=\sum_{(x,y)\in R_c} |U(x,y)|^2$$
作为 logits，经过 softmax 得到分类概率。

## 2. 项目结构（课程规范 / 可复现）

- 代码（核心实现）
	- [src/metasurface_dnn](src/metasurface_dnn)
		- [physics.py](src/metasurface_dnn/physics.py): 角谱法传播
		- [model.py](src/metasurface_dnn/model.py): 3 层相位-only DNN + 探测器
		- [train.py](src/metasurface_dnn/train.py): 训练并导出相位掩膜/指标
		- [simulate.py](src/metasurface_dnn/simulate.py): 生成仿真强度图与摘要
		- [prepare_dataset.py](src/metasurface_dnn/prepare_dataset.py): 把原始图片整理成 `npz`
		- [predict.py](src/metasurface_dnn/predict.py): 对指定 split 输出预测 CSV
		- [export_meta_atoms.py](src/metasurface_dnn/export_meta_atoms.py): 结合 LUT 将相位量化成 meta-atom 几何参数
- 配置
	- [configs/default.yaml](configs/default.yaml)
	- [configs/emnist_letters.yaml](configs/emnist_letters.yaml)
- 数据占位
	- [data/README.md](data/README.md)
- 输出占位
	- [outputs/README.md](outputs/README.md)
- meta-atom LUT 示例
	- [assets/meta_atom_lut_example.csv](assets/meta_atom_lut_example.csv)

注意：仓库中历史参考资料目录 `Diffractive-Deep-Neural-Networks/`、`onn-simulation/` 当前被 `.gitignore` 忽略；本项目的可复现实现在 `src/` 下，不依赖它们被提交。

## 3. 数据集约定（EMNIST Letters / 预留输入输出）

### 3.1 推荐格式（processed）
把处理好的数据放在 `data/processed/`：

- `emnist_letters_train.npz`, `emnist_letters_val.npz`, `emnist_letters_test.npz`

每个 `.npz` 内包含：
- `x`: `(N,H,W)` float32，取值建议归一化到 `[0,1]`（作为入射场幅度）
- `y`: `(N,)` int64，类别标签 `0..25`（对应 26 个字母类别）

### 3.2 自动下载并导出 EMNIST Letters（推荐）

```
python -m metasurface_dnn.prepare_emnist --root data/raw --out_dir data/processed --image_size 128
```

### 3.3 没有数据时
EMNIST 会自动下载并导出到 `data/processed/`，因此通常不需要你手动准备数据。

## 4. 如何运行（训练 + 仿真 + 导出）

### 4.1 安装

```
pip install -r requirements.txt
pip install -e .
```

### 4.2 训练 3 层相位-only DNN（EMNIST Letters）

```
python -m metasurface_dnn.train --config configs/emnist_letters.yaml
```

输出：
- `outputs/<timestamp>/metrics.json`
- `outputs/<timestamp>/phase_masks.npz`
- `outputs/checkpoints/best.pt`

### 4.3 生成仿真强度图（探测面）

```
python -m metasurface_dnn.simulate --config configs/emnist_letters.yaml --checkpoint outputs/checkpoints/best.pt --n 8
```

输出：
- `outputs/simulations/<timestamp>/intensity_sample0.png`
- `outputs/simulations/<timestamp>/batch_inputs.npz`

### 4.4 预测输出 CSV（用于交作业的“预留输出”）

```
python -m metasurface_dnn.predict --config configs/emnist_letters.yaml --checkpoint outputs/checkpoints/best.pt --split test
```

输出：`outputs/predictions/pred_test_<timestamp>.csv`

### 4.5 相位 → meta-atom 几何参数（需要你提供真实 LUT）

题目中“Detailed meta-atom structure is provided”通常意味着你有一个从相位到几何参数（如半径/宽度/高度/旋转角等）的映射表（LUT）。

本仓库提供了一个 LUT 示例文件占位：
- [assets/meta_atom_lut_example.csv](assets/meta_atom_lut_example.csv)

你需要用真实 LUT 替换它后运行：

```
python -m metasurface_dnn.export_meta_atoms \
	--phase_npz outputs/<timestamp>/phase_masks.npz \
	--lut_csv assets/meta_atom_lut_example.csv \
	--out_dir outputs/meta_atom_exports
```

## 5. 你需要按实验参数修改的地方（微波频段）

以 [configs/emnist_letters.yaml](configs/emnist_letters.yaml) 为主：
	- `physics.wavelength_m`: 实验频段对应波长
	- `physics.pixel_size_m`: 采样间隔（等效 meta-atom pitch / 网格间距）
	- `physics.z_list_m`: 4 段传播距离（输入→L1→L2→L3→探测面）
	- `classifier.grid`: 探测窗口网格（26 类默认 2×13），可按探测面尺寸调整

## 6. 对接 3D 打印 / CST 的导出

训练得到的 `phase_masks.npz` 是每层的相位分布（0..2π）。如果你要用均匀介质“厚度变化”来等效相位延迟，可导出厚度图（用于后续建模/3D 打印/CST）：

```
python -m metasurface_dnn.phase_to_thickness \
  --phase_npz outputs/<timestamp>/phase_masks.npz \
  --wavelength_m 3e-2 \
  --n_material 1.6 \
  --out_dir outputs/printable
```

输出为每层的 `*.thickness_m.npy`（单位米）。

## 7. 交付物建议（报告/截图）

- 训练收敛曲线：`metrics.json` 中的 loss/acc（可自行画图）
- 3 层相位分布：`phase_masks.npz`（可视化成 heatmap）
- 探测面强度分布：`intensity_sample0.png`
- 测试集预测表：`pred_test_<timestamp>.csv`

## 8. 32×32 分辨率版本（方便 CST 实验）

在 CST 中如果只想用 32×32 的孔径（减少网格数量），可以用一个单独的 32×32 配置：`configs/emnist_letters_32.yaml`。整体流程与 128×128 类似，只是分辨率、像素间距和输出目录不同。

### 8.1 生成 32×32 EMNIST 数据集

```bash
python -m metasurface_dnn.prepare_emnist \
	--image_size 32 \
	--out_dir data/processed_32
```

生成：
- `data/processed_32/emnist_letters_train.npz`
- `data/processed_32/emnist_letters_val.npz`
- `data/processed_32/emnist_letters_test.npz`

### 8.2 使用 32×32 配置训练

```bash
python -m metasurface_dnn.train --config configs/emnist_letters_32.yaml
```

说明：
- `configs/emnist_letters_32.yaml` 中：
	- 图像分辨率变为 32×32；
	- `physics.pixel_size_m = 4.0e-3` → 32×4 mm = 128 mm，总孔径仍为 128 mm；
	- 探测器网格改为适配 32×32 的 2×13 小方块（`square=2, margin=3`）。
- 输出目录改为 `outputs32/`：
	- `outputs32/<timestamp>/metrics.json`
	- `outputs32/<timestamp>/phase_masks.npz`
	- `outputs32/checkpoints/best.pt`

### 8.3 用 32×32 模型做仿真/预测

仿真探测面强度：

```bash
python -m metasurface_dnn.simulate \
	--config configs/emnist_letters_32.yaml \
	--checkpoint outputs32/checkpoints/best.pt \
	--n 8
```

测试集预测 CSV：

```bash
python -m metasurface_dnn.predict \
	--config configs/emnist_letters_32.yaml \
	--checkpoint outputs32/checkpoints/best.pt \
	--split test
```

### 8.4 32×32 相位 → 厚度（打印/CST 建模）

```bash
python -m metasurface_dnn.phase_to_thickness \
	--phase_npz outputs32/<timestamp>/phase_masks.npz \
	--config configs/emnist_letters_32.yaml \
	--wavelength_m 1.0e-2 \
	--eps_r 2.802 \
	--n0 1.0 \
	--export_csv
```

输出（默认在 `outputs32/printable/` 下）：
- `layer1_phase_rad.thickness_m.npy/.csv`
- `layer2_phase_rad.thickness_m.npy/.csv`
- `layer3_phase_rad.thickness_m.npy/.csv`

### 8.5 生成 32×32 三层结构的 CST 宏

```bash
python scripts/generate_cst_macro.py \
	--config configs/emnist_letters_32.yaml
```

默认会在 `outputs32/cst_macro/` 下生成：
- `build_3layers.cstmacro`

该宏会：
- 读取 `outputs32/printable/` 中 3 层厚度图；
- 在 CST 中按 32×32 像素网格生成 3 个组件 `Layer1/2/3`，像素尺寸为 4 mm，总宽约 128 mm；
- 层间 Z 位置与 `z_list_m = [0.10,0.10,0.10,0.10]` 保持一致。

### 8.6 生成 32×32 字母源宏（CST 波源掩模）

示例：随机选取一张 32×32 测试集字母，生成 32×32 的源掩模和预览图：

```bash
python scripts/generate_cst_source_macro.py \
	--config configs/emnist_letters_32.yaml \
	--resolution 32
```

输出（默认在 `outputs/source_macros/` 下）：
- `source_32_<Letter>_idxrand.cstmacro`
- `source_32_<Letter>_idxrand.png`

说明：
- 该宏在 `z ≈ 0` 平面上画出一个字母形状的 PEC 掩模，孔径大小约 128 mm×128 mm；
- 在 CST 中配合一个从 −z 方向入射、朝 +z 传播的平面波，即可得到“字母形波前”。

### 8.7 生成 32×32 探测器宏（2×13 汇聚区域）

```bash
python scripts/generate_cst_detector_macro.py \
	--config configs/emnist_letters_32.yaml \
	--downsample 1 \
	--out_macro outputs/detector_macros/detector_32.cstmacro
```

该宏会在探测平面 `z ≈ 0.40 m` 处生成 26 个极薄砖块：
- 名称为 `Det_A`…`Det_Z`，对应 26 个字母；
- 在 32×32 网格上布置 2×13 个 2×2 像素的小方块，总宽约 128 mm；
- 与 `configs/emnist_letters_32.yaml` 中的 `classifier.grid` 完全对应，可在 CST 后处理中按区域积分 |E|²。 