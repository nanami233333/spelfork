import os
import shutil

from ase.db import connect
from ase.calculators.vasp import Vasp
from ase.io import write, read
def check_vasp_success(outcar_path):
    """
    检查 VASP 计算是否正常完成，并（可选地）判断是否存在虚频。
    返回：
      - success (bool): 是否正常完成几何优化并成功写出 OUTCAR
      - message (str): 详细说明信息（例如：“正常结束”、“未收敛”等）
      - has_imaginary_freq (bool): 是否检测到虚频（若未进行频率计算，则为 None）
    """
    if not os.path.exists(outcar_path):
        return False, "OUTCAR 文件未找到", None

    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 1) 判断几何优化是否正常结束：检查 OUTCAR 中的“reached required accuracy”
    if "reached required accuracy" not in content:
        return False, "几何优化未正常收敛", None

    # 2) 判断电子自洽是否正常结束：检查 "Gamma" 或 "LOOP+" 中是否有 “reached required accuracy” 电子循环
    #    （此处只做简略示例，若要精确可写更复杂的正则解析）
    if "reached required accuracy for <e>" not in content:
        return False, "电子自洽未正常完成", None

    # 3) 如果脚本配置了 IBRION=5/IBRION=6 做频率计算，则可以在 OUTCAR 中找到频率部分
    #    VASP 在 OUTCAR 中输出振动频率时，行首通常类似 "THz      du/dx" 或 "cm-1  f ="
    #    这里通过查找 “THz” 或 “cm-1” 来判断是否进行过频率计算
    if "THz" in content or "cm-1" in content:
        # 提取所有数值字段，判断是否有小于零的频率 (即虚频)
        freqs = []
        for line in content.splitlines():
            # 例：某些行像 "  1   1   1  0.0000 THz  0.000 cm-1"
            if "THz" in line and "cm-1" in line:
                parts = line.strip().split()
                # 假设倒数第二列是 cm-1 数值
                try:
                    idx = parts.index("cm-1")
                    value = float(parts[idx - 1])
                    freqs.append(value)
                except (ValueError, IndexError):
                    continue
        has_imag = any(f < 0 for f in freqs)
        return True, "优化并有频率信息", has_imag
    else:
        # 未发现频率输出，则只认为几何优化成功，频率检查结果置 None
        return True, "几何优化成功，但未进行频率计算", None
def run_vasp_calculation(atoms, model_name, model_path, vasp_params, do_frequency):
    """
    在指定目录下为给定的 Atoms 对象生成 VASP 输入文件，提交 VASP 计算，并返回计算状态。
    参数：
      - atoms (ase.Atoms)：待计算的原子体系
      - model_name (str)：体系名称，用于文件命名和数据库标识
      - model_path (str)：计算目录，应已存在
      - vasp_params (dict)：包含 INCAR 中的关键参数，例如 ENCUT、ISMEAR、SIGMA、EDIFF、NSW 等
      - do_frequency (bool)：是否在几何优化结束后进行频率计算（IBRION=5/6）
    返回：
      - success (bool)：几何优化是否正常完成
      - message (str)：简要说明
      - has_imaginary_freq (bool 或 None)：是否存在虚频（仅当 do_frequency=True 时有效）
      - optimized_atoms (ase.Atoms)：优化后得到的原子坐标（若优化失败，则返回原始 atoms）
    """
    # 1. 准备 POSCAR: 直接使用 ase.io.write 将 Atoms 对象写到 model_path/ POSCAR
    poscar_path = os.path.join(model_path, "POSCAR")
    write(poscar_path, atoms, format="vasp")

    # 2. 准备 POTCAR: 假设环境变量 VASP_PSP 指向了伪势所在目录，按元素顺序依次拼接
    #    ASE Vasp 计算器会自动寻找 POTCAR，但也可以手动复制。
    #    这里示例不手动写 POTCAR，而由 ASE Vasp 计算器处理。
    # 3. 生成 INCAR: 根据 vasp_params 写入关键字
    incar_lines = []
    incar_lines.append(f"SYSTEM = {model_name}")
    for key, val in vasp_params.items():
        incar_lines.append(f"{key} = {val}")
    # 如果需要频率计算，则在优化完成后重写 INCAR，设置 IBRION=5 或 6，并将 NSW=0
    if do_frequency:
        # 先进行几何优化：默认 IBRION=2（共轭梯度），NSW=N（离子步数）
        incar_lines_opt = incar_lines.copy()
        incar_lines_opt.append("IBRION = 2")
        incar_lines_opt.append(f"NSW = {vasp_params.get('NSW', 50)}")
        incar_lines_opt.append("ISMEAR = 0")
        incar_lines_opt.append("SIGMA = 0.05")
        # 将该 INCAR 写到 model_path/INCAR
        with open(os.path.join(model_path, "INCAR"), 'w') as f:
            f.write("\n".join(incar_lines_opt) + "\n")
    else:
        # 只做一次几何优化或 single-point 计算
        with open(os.path.join(model_path, "INCAR"), 'w') as f:
            f.write("\n".join(incar_lines) + "\n")

    # 4. 生成 KPOINTS: 这里示例使用一个简单的 3×3×3 Monkhorst-Pack 网格
    kpoints_path = os.path.join(model_path, "KPOINTS")
    with open(kpoints_path, 'w') as f:
        f.write("Automatic mesh\n")
        f.write("0\n")
        f.write("Monkhorst-Pack\n")
        f.write("3 3 3\n")
        f.write("0 0 0\n")

    # 5. 调用 ASE Vasp 计算器提交几何优化
    calc = Vasp(directory=model_path,
                # 指定用到的 POTCAR 可以通过 ASE 自动查找，也可以在外部环境配置
                kpts=[3, 3, 3],
                **vasp_params  # 包括 ENCUT, EDIFF, ISMEAR, SIGMA, NELM, NSW 等
                )
    atoms.set_calculator(calc)

    optimized_atoms = atoms.copy()
    try:
        # 5.1 执行几何优化：这一步会调用 VASP 进行离子步迭代
        energy = optimized_atoms.get_potential_energy()
    except Exception as e:
        print(f"Error during VASP geometry optimization: {str(e)}")
        return False, "VASP 优化过程中出现异常", None, atoms

    # 5.2 几何优化结束后，从 CONTCAR 中读取优化后的坐标（若 VASP 正常生成）
    contcar_path = os.path.join(model_path, "CONTCAR")
    if os.path.exists(contcar_path):
        optimized_atoms = read(contcar_path, format="vasp")  # 读取优化后体系
    else:
        # 若未生成 CONTCAR，直接使用当前 optimized_atoms
        print("未检测到 CONTCAR，可能优化未产生新的结构。")
        optimized_atoms = optimized_atoms

    # 6. 进行频率计算（可选）：在优化完成后重写 INCAR，将 IBRION 设置为 5/6
    has_imag = None
    if do_frequency:
        # 6.1 重写 INCAR 做频率：设置 IBRION=5(或6)、NFREE=2、NSW=0 并保持其余与几何优化相同
        incar_freq = incar_lines.copy()
        incar_freq.append("IBRION = 5")
        incar_freq.append("NFREE = 2")
        incar_freq.append("NSW = 0")
        with open(os.path.join(model_path, "INCAR"), 'w') as f:
            f.write("\n".join(incar_freq) + "\n")
        # 6.2 用新的计算器重置
        calc_freq = Vasp(directory=model_path,
                         kpts=[3, 3, 3],
                         **vasp_params,
                         IBRION=5, NFREE=2, NSW=0
                         )
        optimized_atoms.set_calculator(calc_freq)
        try:
            optimized_atoms.get_potential_energy()  # 触发频率（静态）计算
        except Exception as e:
            print(f"Error during VASP frequency calculation: {str(e)}")
            # 此时认为几何优化虽成功，但频率计算失败
            success, message, _ = check_vasp_success(os.path.join(model_path, "OUTCAR"))
            return success, message, None, optimized_atoms

        # 6.3 读取频率判断
        success, message, has_imag = check_vasp_success(os.path.join(model_path, "OUTCAR"))
    else:
        # 只做几何优化，不主动做频率
        success, message, has_imag = check_vasp_success(os.path.join(model_path, "OUTCAR"))

    return success, message, has_imag, optimized_atoms
# 连接各个数据库文件（若不存在则自动创建）
db_initial = connect('initial_db.db')          # 待计算的初始结构
db_optimized = connect('optimized_vasp.db')    # 优化成功且无虚频
db_imag = connect('imaginary_freq_vasp.db')    # 优化成功但含虚频
db_nonconv = connect('nonconverged_vasp.db')   # 优化/频率未收敛
db_error = connect('error_vasp.db')            # 计算过程中出现异常
# VASP 全局计算参数示例
vasp_params = {
    'ENCUT': 400,        # 平面波截断能 (eV)
    'EDIFF': 1e-5,       # 电子收敛阈值
    'ISMEAR': 0,         # 0: Gauss 展宽，用于绝缘体/分子
    'SIGMA': 0.05,       # 展宽宽度 (eV)
    'IBRION': 2,         # 优化算法：CG
    'NSW': 50,           # 离子迭代步数
    'ISPIN': 1,          # 自旋极化：1 表示不考虑自旋
    'LCHARG': True,      # 写 CHGCAR 以便后续分析
    'LWAVE': False       # 不写 WAVECAR 以节省磁盘
}
calc_dir = os.getcwd()                   # 当前脚本所在目录，当作所有计算的根目录
os.makedirs(calc_dir, exist_ok=True)     # 确保目录存在

for row in db_initial.select():
    atoms = row.toatoms()                # 从记录中取出 ASE Atoms 对象
    model_name = row.name or row.id      # 用于标识体系的名称（可从记录字段获取）

    # 1. 检查该 model_name 是否已经出现在其他库中，若已记录则跳过
    already_done = (
        db_optimized.count(name=model_name) or
        db_imag.count(name=model_name) or
        db_nonconv.count(name=model_name) or
        db_error.count(name=model_name)
    )
    if already_done:
        continue

    # 2. 为当前体系创建专属文件夹，用于存放 VASP 所有输入/输出
    model_path = os.path.join(calc_dir, model_name)
    os.makedirs(model_path, exist_ok=True)

    try:
        # 3. 提交 VASP 计算，先几何优化，后可选频率
        success, message, has_imag, optimized_atoms = run_vasp_calculation(
            atoms, model_name, model_path, vasp_params, do_frequency=True
        )

        # 4. 根据计算结果将优化后的数据写入相应数据库
        if success:
            if has_imag is True:
                # 优化成功但含虚频
                db_imag.write(optimized_atoms, name=model_name)
            else:
                # 优化成功且无虚频
                db_optimized.write(optimized_atoms, name=model_name)
        else:
            # 优化或频率计算未收敛
            db_nonconv.write(optimized_atoms, name=model_name)

    except Exception as e:
        # 5. 捕获脚本运行过程中任何意外异常
        print(f"Error processing {model_name}: {str(e)}")
        # 尝试读取 CONTCAR 作为 fallback
        contcar_path = os.path.join(model_path, "CONTCAR")
        if os.path.exists(contcar_path):
            fallback_atoms = read(contcar_path, format="vasp")
        else:
            fallback_atoms = atoms
        # 把失败数据写入 error 库
        db_error.write(fallback_atoms, name=model_name)

    finally:
        # 6. 验证输出文件是否生成，并打印摘要信息（可选）
        expected_files = ["POSCAR", "INCAR", "KPOINTS", "POTCAR", "CONTCAR", "OUTCAR"]
        for fname in expected_files:
            if os.path.exists(os.path.join(model_path, fname)):
                print(f"{model_name}/{fname} 已生成")
            else:
                print(f"{model_name}/{fname} 未找到")
