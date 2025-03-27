import os
import shutil
from ase.db import connect
from ase.calculators.gaussian import Gaussian
from ase.io import read  # Using ase.io.read instead of read_gaussian_out

def check_calculation_success(log_path, chk_path):
    """
    Comprehensively check if the Gaussian calculation completed successfully.
    
    Returns:
    - success (bool): Whether the calculation succeeded
    - message (str): Detailed message
    - has_imaginary_freq (bool): Whether imaginary frequencies exist
    """
    if not os.path.exists(log_path):
        return False, "Log file not found", None

    with open(log_path, 'r') as file:
        content = file.read()
        
    # Check if terminated normally
    normal_termination = "Normal termination" in content
    if not normal_termination:
        return False, "Gaussian did not terminate normally", None

    # Check if optimization completed
    optimization_completed = "Stationary point found" in content
    if not optimization_completed:
        return False, "Optimization did not complete successfully", None

    # Check frequency calculation
    freq_completed = "Harmonic frequencies" in content
    if not freq_completed:
        return False, "Frequency calculation did not complete", None

    # Check for imaginary frequencies
    has_imaginary_freq = False
    freq_sections = content.split("Frequencies --")
    frequencies = []
    for section in freq_sections[1:]:
        lines = section.strip().split("\n")
        if lines:
            freqs_in_line = [float(f) for f in lines[0].split()]
            frequencies.extend(freqs_in_line)
    has_imaginary_freq = any(f < 0 for f in frequencies)

    return True, "Calculation completed successfully", has_imaginary_freq

def calculate_multiplicity(atoms):
    """Calculate the electronic multiplicity of the system"""
    total_electrons = sum(atom.number for atom in atoms)
    return 2 if total_electrons % 2 != 0 else 1

def run_gaussian_calculation(atoms, model_name, model_path, params, charge, multiplicity):
    """Run the Gaussian calculation and return whether it was successful"""
    input_filename = os.path.join(model_path, f"{model_name}_opt_freq.com")
    chk_filename = os.path.join(model_path, params['chk'])  # Using chk parameter from configuration dictionary
    log_filename = os.path.join(model_path, f"{model_name}_opt_freq.log")
    
    with open(input_filename, 'w') as f:
        f.write(f"%chk={chk_filename}\n")
        f.write(f"%mem={params['mem']}\n")
        f.write(f"%nprocshared={params['nprocshared']}\n")
        f.write("#P\n")
        f.write(f"{params['method']}\n")
        f.write(f"opt({params['opt']})\n")
        
        # Process freq parameter
        freq_value = params['freq']
        if freq_value.lower() == 'freq':
            f.write("freq\n")
        else:
            f.write(f"freq={freq_value}\n")
        
        # Process polar parameter
        polar_value = params['polar']
        if polar_value.lower() == 'polar':
            f.write("polar=Opt\n")
        else:
            f.write(f"polar={polar_value}\n")
        
        f.write("\n")
        f.write("Gaussian input prepared by ASE\n\n")
        f.write(f"{charge} {multiplicity}\n")
        for atom in atoms:
            f.write(f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n")
        f.write("\n")

    calc = Gaussian(label=os.path.join(model_path, model_name + '_opt_freq'), **params)
    atoms.set_calculator(calc)
    try:
        atoms.get_potential_energy()
    except Exception as e:
        print(f"Error during Gaussian calculation: {str(e)}")
    
    # Verify if the log file exists
    if os.path.exists(log_filename):
        print(f"Log file found: {log_filename}")
    else:
        print(f"Log file not found: {log_filename}")
    
    # Attempt to use ase.io.read to read updated geometry
    try:
        updated_atoms = read(log_filename, format='gaussian-out')
        if updated_atoms is not None:
            print("Updated atoms read successfully using ase.io.read.")
            atoms = updated_atoms
        else:
            print("Failed to read updated atoms using ase.io.read.")
    except Exception as read_e:
        print(f"Error reading Gaussian output file with ase.io.read: {str(read_e)}")
        import traceback
        traceback.print_exc()
    
    success, message, has_imaginary_freq = check_calculation_success(log_filename, chk_filename)
    return success, message, has_imaginary_freq, atoms  # Return the updated atoms

# Create database connections
db = connect('initial_db.db')
optimized_db = connect('optimized_db.db')
nonconverged_db = connect('nonconverged.db')
error_db = connect('error.db')
imaginary_freq_db = connect('imaginary_freq.db')

# Low precision optimization and frequency calculation parameters
low_precision_params = {
    'mem': '40GB',
    'nprocshared': 40,
    'method': 'B3LYP/3-21G',
    'opt': 'loose,MaxCycle=1000',
    'freq': 'freq',
    'polar': 'polar',
    'chk': ''  # This will be dynamically set based on the model name
}

calc_dir = os.getcwd()
os.makedirs(calc_dir, exist_ok=True)

for row in db.select():
    atoms = row.toatoms()
    model_name = row.formula

    # Check if the model is already in any of the databases
    is_existing_model = (
        optimized_db.count(formula=model_name) != 0 or
        nonconverged_db.count(formula=model_name) != 0 or
        error_db.count(formula=model_name) != 0 or
        imaginary_freq_db.count(formula=model_name) != 0
    )

    if not is_existing_model:
        # Dynamically set .chk file name
        low_precision_params['chk'] = f"{model_name}_opt_freq.chk"

        # Create folder only if calculation is needed
        model_path = os.path.join(calc_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        try:
            charge = 0  # Assume charge is 0; modify if necessary
            multiplicity = calculate_multiplicity(atoms)
            
            # Perform low precision optimization and frequency calculation
            success, message, has_imaginary_freq, atoms = run_gaussian_calculation(
                atoms, model_name, model_path, low_precision_params, charge, multiplicity)
            
            if success:
                if has_imaginary_freq:
                    print(f"{model_name}: Calculation successful but has imaginary frequencies")
                    imaginary_freq_db.write(atoms, model_name=model_name)
                else:
                    print(f"{model_name}: Calculation successful")
                    optimized_db.write(atoms, model_name=model_name)
            else:
                print(f"{model_name}: {message}")
                nonconverged_db.write(atoms, model_name=model_name)
        except Exception as e:
            print(f"Error occurred with {model_name}: {str(e)}")
            # Read updated geometry from output file
            try:
                log_filename = os.path.join(model_path, f"{model_name}_opt_freq.log")
                if os.path.exists(log_filename):
                    updated_atoms = read(log_filename, format='gaussian-out')
                    if updated_atoms is not None:
                        atoms = updated_atoms
            except Exception as read_e:
                print(f"Error reading Gaussian output file: {str(read_e)}")
            try:
                error_db.write(atoms, model_name=model_name)
            except Exception as db_e:
                print(f"Error writing to error_db: {str(db_e)}")
        finally:
            # Manage output files
            output_files = [f"{model_name}_opt_freq.chk", f"{model_name}_opt_freq.log", f"{model_name}_opt_freq.com"]
            for filename in output_files:
                file_path = os.path.join(model_path, filename)
                if os.path.exists(file_path):
                    print(f"File exists: {file_path}")
                else:
                    print(f"File does not exist: {file_path}")
    else:
        print(f"Skipping task for {model_name} as it's already in one of the databases.")
