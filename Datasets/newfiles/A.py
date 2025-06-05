import os
import glob

def convert_mol_to_poscar(input_pattern="molecule_*.mol",
                          output_dir="output",
                          margin=15.0):
    """
    将匹配 input_pattern 的 .mol 文件，转换为 VASP 可用的 POSCAR 格式。
    1) margin：在 x, y, z 三个方向各自的最小–最大范围上，加的真空层 (Å)。
    2) 元素顺序：['O','C','H','N','F']。如果你的 .mol 里包含其他元素，请自行扩展此列表，并保持写入顺序一致。
    """

    # —— STEP 0: 准备输出目录 —— #
    os.makedirs(output_dir, exist_ok=True)

    # —— STEP 1: 定义 POSCAR 中的元素顺序 —— #
    elements = ["O", "C", "H", "N", "F", "Cl"]
    # 创建一个方便检查的“空列表”，确保后面能按元素顺序往里塞坐标
    # coords_by_elem['O'] = [(x1,y1,z1), (x2,y2,z2), ...]
    coords_by_elem = {el: [] for el in elements}

    # —— STEP 2: 遍历所有 .mol 文件 —— #
    for filepath in glob.glob(input_pattern):
        # 2.1 先清空字典与坐标记录
        for el in elements:
            coords_by_elem[el] = []
        all_x, all_y, all_z = [], [], []  # 用来计算最小/最大

        # 2.2 逐行读取 .mol（假设前 4 行是 header，从第 5 行开始是真正的“x y z 元素 0 0 0...”）
        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines[4:]:
            s = line.strip()
            if s.startswith("M"):
                # 遇到 “M  END” 这样的行就停止原子读取
                break
            parts = s.split()
            if len(parts) >= 4:
                try:
                    x_f = float(parts[0])
                    y_f = float(parts[1])
                    z_f = float(parts[2])
                    el = parts[3]
                except ValueError:
                    # 如果本行不是 “数值 数值 数值 元素”，就跳过
                    continue

                if el in coords_by_elem:
                    coords_by_elem[el].append((x_f, y_f, z_f))
                    all_x.append(x_f)
                    all_y.append(y_f)
                    all_z.append(z_f)
                else:
                    # 如果 .mol 里出现了不在 ['O','C','H','N','F'] 里的元素，会被忽略
                    # 你可以在 elements 里自行添加所需元素并保持顺序
                    print(f"⚠️   文件 {filepath} 中发现未列入 {elements} 的元素 '{el}'，已忽略。")

        # 如果读不到任何原子，跳过该文件
        if len(all_x) == 0:
            print(f"⚠️   文件 {filepath} 中没有找到任何支持的原子，跳过。")
            continue

        # —— STEP 3: 计算分子整体最小/最大坐标并生成“正交晶胞向量” —— #
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)

        # 根据范围加上 margin，得到最终晶胞的边长
        Lx = (x_max - x_min) + margin
        Ly = (y_max - y_min) + margin
        Lz = (z_max - z_min) + margin

        # —— STEP 4: 把每个原子“居中”到晶胞中 —— #
        # VASP POSCAR 中，如果写 Cartesian 坐标，就要保证所有点都在 [0, Lx] × [0, Ly] × [0, Lz] 范围里
        # 我们先把原子坐标从 [x_min, x_max] 映射到 [0, x_max-x_min]，然后再在每个方向加上 margin/2，让分子整体居中
        # new_x = (x_f - x_min) + margin/2
        shifted_by_elem = {el: [] for el in elements}
        half_margin = margin / 2.0
        for el in elements:
            for (x_f, y_f, z_f) in coords_by_elem[el]:
                new_x = (x_f - x_min) + half_margin
                new_y = (y_f - y_min) + half_margin
                new_z = (z_f - z_min) + half_margin
                # 简单验证：new_x ∈ [ half_margin, (x_max - x_min) + half_margin ] = [ half_margin, Lx - half_margin ]
                # 这样所有 new_x 均 ∈ [0, Lx]（边缘处没有 atom 贴着 0 或 Lx）
                shifted_by_elem[el].append((new_x, new_y, new_z))

        # —— STEP 5: 开始拼接 POSCAR 文件的各行内容 —— #
        poscar_lines = []

        # (1) 第一行：注释，可留空
        name = os.path.splitext(os.path.basename(filepath))[0]  # “molecule_123”
        poscar_lines.append(f"# Converted from {os.path.basename(filepath)}")

        # (2) 第二行：缩放因子，通常写 1.0
        poscar_lines.append("1.00000000000000")

        # (3)(4)(5) 三个正交晶胞向量
        poscar_lines.append(f"{Lx: .6f}  0.000000  0.000000")
        poscar_lines.append(f"0.000000  {Ly: .6f}  0.000000")
        poscar_lines.append(f"0.000000  0.000000  {Lz: .6f}")

        # (6) 元素顺序
        poscar_lines.append(" ".join(elements))

        # (7) 各元素的个数（顺序要和第 6 行一致）
        counts = [len(coords_by_elem[el]) for el in elements]
        poscar_lines.append(" ".join(str(c) for c in counts))

        # (8) 坐标类型：Cartesian，一定要写！
        poscar_lines.append("Cartesian")

        # (9) 坐标：必须“先把所有 O 写完，再把所有 C 写完……”
        for el in elements:
            for (new_x, new_y, new_z) in shifted_by_elem[el]:
                poscar_lines.append(f"{new_x: .6f}  {new_y: .6f}  {new_z: .6f}  {el}")

        # —— STEP 6: 写文件 —— #
        out_path = os.path.join(output_dir, f"{name}.POSCAR")
        with open(out_path, "w") as fout:
            fout.write("\n".join(poscar_lines))

        print(f"✅  {filepath} → {out_path} (box = [{Lx:.2f}, {Ly:.2f}, {Lz:.2f}] Å)")

    print("全部转换完毕。")

if __name__ == "__main__":
    # margin 可以根据需要自己调大/调小。例如一开始设为 10.0、12.0、15.0 都行，如果分子较密集，建议 15–20 Å。
    convert_mol_to_poscar(input_pattern="molecule_*.mol", output_dir="output", margin=15.0)
