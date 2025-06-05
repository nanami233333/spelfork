import os
import glob

def convert_mol_to_poscar(input_pattern="molecule_*.mol",
                          output_dir="output",
                          margin=10.0):
    """
    将符合 input_pattern 的所有 .mol 文件转换为 POSCAR 格式，并写到 output_dir 下。
    - margin: 在 x、y、z 三个方向的原子最小/最大值区间上，各额外加的真空层 (Å 单位)。
    """
    # 指定 POSCAR 中元素的顺序
    elements = ["O", "C", "H", "N", "F"]
    os.makedirs(output_dir, exist_ok=True)

    for filepath in glob.glob(input_pattern):
        coords = []           # 存放 (x,y,z,元素) 的列表，x,y,z 为 float
        elem_counts = {el: 0 for el in elements}

        with open(filepath, "r") as f:
            lines = f.readlines()

        # 从第五行（索引 4）开始读取原子数据，遇到以“M”开头的行就停止
        for line in lines[4:]:
            stripped = line.strip()
            if stripped.startswith("M"):
                break

            parts = stripped.split()
            # mol 文件一行通常是：x y z 元素 以及若干 0
            # 我们只取前 4 列：x, y, z, 元素
            if len(parts) >= 4:
                x_f = float(parts[0])
                y_f = float(parts[1])
                z_f = float(parts[2])
                el = parts[3]
                coords.append((x_f, y_f, z_f, el))
                if el in elem_counts:
                    elem_counts[el] += 1

        if len(coords) == 0:
            # 如果这个文件里没读取到原子，就跳过
            print(f"注意：文件 {filepath} 没有读取到任何原子，已跳过。")
            continue

        # 计算 xyz 三个方向的最小值和最大值
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        zs = [p[2] for p in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        # 计算晶胞尺寸：在最大–最小范围上加上前后各 margin/2
        # 也可以直接 Lx = (x_max - x_min) + margin
        Lx = (x_max - x_min) + margin
        Ly = (y_max - y_min) + margin
        Lz = (z_max - z_min) + margin

        # 构造 POSCAR 内容
        poscar_lines = []

        # 第一行：留空或写一个注释性标题
        poscar_lines.append("")  # 你也可以写成：f"# {os.path.basename(filepath)}"

        # 第二行：缩放因子（一般写 1.0）
        poscar_lines.append("1.00000000000000")

        # 第三~第五行：三个晶格向量（假设正交直角晶胞）
        # 这里我们直接把向量写成 (Lx,0,0), (0,Ly,0), (0,0,Lz)
        poscar_lines.append(f"{Lx:.6f} 0.000000 0.000000")
        poscar_lines.append(f"0.000000 {Ly:.6f} 0.000000")
        poscar_lines.append(f"0.000000 0.000000 {Lz:.6f}")

        # 第六行：元素符号，顺序和前面 elem_counts 一致
        poscar_lines.append(" ".join(elements))

        # 第七行：对应元素的个数
        counts_line = " ".join(str(elem_counts[el]) for el in elements)
        poscar_lines.append(counts_line)

        # 接下来写入原子坐标：直接用 Cartesian 坐标 (Direct/Cartesian) 模式
        # 如果要写成 Direct 坐标，需要把 (x, y, z) 坐标映射到 0~1 之间，这里示例直接写成 Cartesian
        poscar_lines.append("Cartesian")
        for x_f, y_f, z_f, el in coords:
            # 偏移：如果你希望分子居中，可以把原子坐标先减去 ( (x_min + x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2 )
            # 然后再加上 (Lx/2, Ly/2, Lz/2)，保证分子在晶胞中央。下面示例做了“居中”：
            x_centered = x_f - (x_min + x_max) / 2.0 + Lx / 2.0
            y_centered = y_f - (y_min + y_max) / 2.0 + Ly / 2.0
            z_centered = z_f - (z_min + z_max) / 2.0 + Lz / 2.0
            poscar_lines.append(f"{x_centered:.6f} {y_centered:.6f} {z_centered:.6f} {el}")

        # 写入文件
        filename = os.path.basename(filepath)
        name, _ = os.path.splitext(filename)
        out_path = os.path.join(output_dir, f"{name}.POSCAR")
        with open(out_path, "w") as fout:
            fout.write("\n".join(poscar_lines))

        print(f"已生成：{out_path}")

if __name__ == "__main__":
    # 你可以修改 margin 的值，比如 margin=8.0 或 12.0
    convert_mol_to_poscar(input_pattern="molecule_*.mol", output_dir="output", margin=10.0)