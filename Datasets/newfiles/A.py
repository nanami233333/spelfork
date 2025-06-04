import os
import glob

def convert_mol_to_poscar(input_pattern="molecule_*.mol", output_dir="output"):
    """
    遍历所有符合 input_pattern 的 .mol 文件，将其转换为 POSCAR 格式，
    并将结果保存到 output_dir 目录下，文件名为原文件名对应的 .POSCAR。
    """
    # 要包含的元素顺序
    elements = ["O", "C", "H", "N", "F"]
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    for filepath in glob.glob(input_pattern):
        # 初始化坐标列表与元素计数字典
        coords = []
        elem_counts = {el: 0 for el in elements}

        with open(filepath, "r") as f:
            lines = f.readlines()

        # 跳过前 4 行（注释行、程序信息行、空行、计数行），从第 5 行开始读取原子数据
        for line in lines[4:]:
            stripped = line.strip()
            if stripped.startswith("M"):
                # 遇到 "M  END" 就停止读取原子部分
                break
            parts = stripped.split()
            if len(parts) >= 4:
                x, y, z, el = parts[0], parts[1], parts[2], parts[3]
                coords.append((x, y, z, el))
                if el in elem_counts:
                    elem_counts[el] += 1

        # 构造 POSCAR 内容
        poscar_lines = []

        # 第一行留空
        poscar_lines.append("")

        # 第二行：缩放因子
        poscar_lines.append("1.000")

        # 第三到第五行：晶格向量（此处全用 0.0 0.0 0.0）
        for _ in range(3):
            poscar_lines.append("0.0 0.0 0.0")

        # 第六行：元素符号，按 O C H N F 顺序
        poscar_lines.append(" ".join(elements))

        # 第七行：对应元素的计数
        counts_line = " ".join(str(elem_counts[el]) for el in elements)
        poscar_lines.append(counts_line)

        # 从第八行开始：保留所有原子坐标，按输入文件中的顺序，格式为 “x y z”
        for x, y, z, _ in coords:
            poscar_lines.append(f"{x} {y} {z}")

        # 生成输出文件路径，并写入内容
        filename = os.path.basename(filepath)
        name, _ = os.path.splitext(filename)
        out_path = os.path.join(output_dir, f"{name}.POSCAR")
        with open(out_path, "w") as fout:
            fout.write("\n".join(poscar_lines))

if __name__ == "__main__":
    convert_mol_to_poscar()
