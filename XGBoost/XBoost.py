import pandas as pd
import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.local_env import VoronoiNN, JmolNN
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import glob
import os
import psutil
import gc
import csv
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.local_env import OpenBabelNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import shap  # SHAP explanation

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def load_data(cif_file_path, y_data_path, homo_lumo_path):
    structures = []
    y_values = pd.read_csv(y_data_path, index_col=0)
    y_values = y_values.select_dtypes(include=[np.number])
    
    homo_lumo_data = pd.read_csv(homo_lumo_path, index_col=0)
    
    for cif_file in glob.glob(cif_file_path):
        structure = Structure.from_file(cif_file)
        structures.append(structure)
    
    min_length = min(len(structures), len(y_values))
    structures = structures[:min_length]
    y_values = y_values.iloc[:min_length]
    
    # Merge y_values with homo_lumo_data
    combined_data = pd.concat([y_values, homo_lumo_data], axis=1)
    
    return structures, combined_data

def extract_features(structures, homo_lumo_data):
    features = []
    
    # Electronegativities and other elemental properties
    electronegativity = {el.symbol: el.X for el in Element}
    atomic_radii = {el.symbol: el.atomic_radius for el in Element}
    
    # Manually define polarizability values
    polarizability_values = {
        'H': 0.6668,
        'C': 1.76,
        'N': 1.10,
        'O': 0.802,
        'F': 0.557,
        'Cl': 2.18,
        'Br': 3.05,
        'I': 5.35,
        'S': 2.90,
        'P': 3.63,
        'Si': 5.38
    }
    
    # Define polar atoms for polar surface area calculation
    polar_atoms = ['O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'P']
    
    # Set neighbor cutoff radius (in Å)
    neighbor_cutoff = 1.2  # Adjust as appropriate for your molecules
    
    for i, structure in enumerate(structures):
        if i >= len(homo_lumo_data):
            break
        
        # Convert structure to molecule (assuming isolated molecule)
        molecule = Molecule.from_sites(structure.sites)
        
        # Basic molecular properties
        num_atoms = len(molecule)
        total_mass = molecule.composition.weight
        avg_atomic_mass = total_mass / num_atoms if num_atoms > 0 else 0
        
        # Atom type fractions
        element_counts = molecule.composition.get_el_amt_dict()
        atom_type_fractions = {el: count / num_atoms for el, count in element_counts.items()}
        
        # Average electronegativity
        avg_electronegativity = np.mean([electronegativity.get(site.specie.symbol, 0) for site in molecule])
        
        # Estimate dipole moment (simplified)
        charges = np.array([electronegativity.get(site.specie.symbol, 0) for site in molecule])
        coords = np.array(molecule.cart_coords)
        dipole_moment_vector = np.sum(charges[:, np.newaxis] * coords, axis=0)
        dipole_moment = np.linalg.norm(dipole_moment_vector)
        
        # Total polarizability
        total_polarizability = sum(polarizability_values.get(site.specie.symbol, 0) for site in molecule)
        
        # Hydrogen bond donors and acceptors
        h_bond_donors = 0
        h_bond_acceptors = 0
        for idx, site in enumerate(molecule):
            symbol = site.specie.symbol
            if symbol == 'H':
                neighbors = molecule.get_neighbors(site, r=neighbor_cutoff)
                if any(neigh.specie.symbol in ['O', 'N', 'F'] for neigh in neighbors):
                    h_bond_donors += 1
            elif symbol in ['O', 'N', 'F']:
                h_bond_acceptors += 1
        
        # Rotatable bonds estimation (simplified)
        rotatable_bonds = 0
        for idx, site in enumerate(molecule):
            symbol = site.specie.symbol
            if symbol != 'H':
                neighbors = molecule.get_neighbors(site, r=neighbor_cutoff)
                non_h_neighbors = [neigh for neigh in neighbors if neigh.specie.symbol != 'H']
                if len(non_h_neighbors) > 1:
                    rotatable_bonds += len(non_h_neighbors) - 1
        
        # Calculate polar surface area (PSA)
        psa = sum(atomic_radii.get(site.specie.symbol, 0) ** 2 for site in molecule if site.specie.symbol in polar_atoms)
        
        # Molecular volume and surface area estimation
        # Simplified estimation using atomic radii
        molecular_volume = sum((4/3) * np.pi * (atomic_radii.get(site.specie.symbol, 0) ** 3) for site in molecule)
        molecular_surface_area = sum(4 * np.pi * (atomic_radii.get(site.specie.symbol, 0) ** 2) for site in molecule)
        
        # Functional group counts (simplified)
        hydroxyl_groups = 0
        for idx, site in enumerate(molecule):
            if site.specie.symbol == 'O':
                neighbors = molecule.get_neighbors(site, r=neighbor_cutoff)
                h_neighbors = [neigh for neigh in neighbors if neigh.specie.symbol == 'H']
                if len(h_neighbors) > 0:
                    hydroxyl_groups += 1
        
        # Feature vector
        feature = [
            num_atoms,
            total_mass,
            avg_atomic_mass,
            avg_electronegativity,
            dipole_moment,
            total_polarizability,
            h_bond_donors,
            h_bond_acceptors,
            rotatable_bonds,
            psa,
            molecular_volume,
            molecular_surface_area,
            hydroxyl_groups
        ]
        
        # Add fractions of common elements
        for element in ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']:
            fraction = atom_type_fractions.get(element, 0)
            feature.append(fraction)
        
        # Add HOMO-LUMO data
        homo = homo_lumo_data.iloc[i].get('HOMO', np.nan)
        lumo = homo_lumo_data.iloc[i].get('LUMO', np.nan)
        homo_lumo_gap = homo_lumo_data.iloc[i].get('HOMO-LUMO_gap', np.nan)
        feature.extend([homo, lumo, homo_lumo_gap])
        
        features.append(feature)
    
    # Define column names
    columns = [
        'num_atoms', 'total_mass', 'avg_atomic_mass', 'avg_electronegativity',
        'dipole_moment', 'total_polarizability', 'h_bond_donors', 'h_bond_acceptors',
        'rotatable_bonds', 'polar_surface_area', 'molecular_volume', 'molecular_surface_area',
        'hydroxyl_groups'
    ]
    
    # Add element fraction columns
    columns.extend([f'{el}_fraction' for el in ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']])
    
    # Add HOMO-LUMO columns
    columns.extend(['HOMO', 'LUMO', 'HOMO-LUMO_gap'])
    
    return pd.DataFrame(features, columns=columns)


def plot_partial_dependence(model, X, features):
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(model, X, features=features, ax=ax)
    plt.tight_layout()
    plt.savefig('partial_dependence.png')
    plt.close()

def plot_shap_summary(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, features=X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, min(y_pred), max(y_pred), colors='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.savefig('residuals_vs_predicted.png')
    plt.close()

def plot_hyperparameter_performance(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    # Create pivot table for heatmap
    pivot_table = results.pivot(index='param_n_estimators', columns='param_max_depth', values='mean_test_score')
    pivot_table = -pivot_table  # Convert negative MSE to positive MSE

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f")
    plt.title('Hyperparameter Grid Search Results (MSE)')
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.tight_layout()
    plt.savefig('hyperparameter_heatmap.png')
    plt.close()

def plot_error_vs_feature(y_test, y_pred, X_test_scaled, feature_names, feature_name):
    errors = y_test - y_pred
    feature_index = feature_names.index(feature_name)
    feature_values = X_test_scaled[:, feature_index]
    plt.figure(figsize=(10, 6))
    plt.scatter(feature_values, errors, alpha=0.5)
    plt.hlines(0, min(feature_values), max(feature_values), colors='red')
    plt.xlabel(feature_name)
    plt.ylabel('Error')
    plt.title(f'Error vs {feature_name}')
    plt.savefig(f'error_vs_{feature_name}.png')
    plt.close()

def plot_correlation_matrix(X_scaled, feature_names):
    corr_matrix = pd.DataFrame(X_scaled, columns=feature_names).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def plot_prediction_distribution(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_test, label='Actual', shade=True)
    sns.kdeplot(y_pred, label='Predicted', shade=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    plt.close()

def train_xgboost_model(X_scaled, y):
    print_memory_usage()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Feature Scaling is already done before calling this function
    # Hyperparameter grid and model setup
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3]
    }

    model = XGBRegressor(random_state=42, tree_method='hist')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    print("Grid search completed.")

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE scores: {rmse_scores}")
    print(f"Average RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")

    # Test set evaluation
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test set RMSE: {rmse:.4f}")
    print(f"Test set Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test set R² Score: {r2:.4f}")

    return best_model, X_test, y_test, y_pred, grid_search

def train_decision_tree_model(X_scaled, y):
    print_memory_usage()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train DecisionTreeRegressor
    tree_model = DecisionTreeRegressor(random_state=42, max_depth=5)  # You can adjust max_depth
    tree_model.fit(X_train, y_train)

    # Test set evaluation
    y_pred = tree_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Decision Tree - Test set RMSE: {rmse:.4f}")
    print(f"Decision Tree - Test set MAE: {mae:.4f}")
    print(f"Decision Tree - Test set R² Score: {r2:.4f}")

    return tree_model, X_test, y_test

def plot_decision_tree(tree_model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree_visualization.png")
    plt.show()
    
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plotting True vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted")
    plt.savefig("true_vs_predicted.png")
    plt.close()

    # Save data to CSV
    pd.DataFrame({
        'True_Values': y_test,
        'Predicted_Values': y_pred
    }).to_csv('true_vs_predicted.csv', index=False)


def plot_learning_curve(model, X_scaled, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error')

    train_scores_mean = np.sqrt(-np.mean(train_scores, axis=1))  # RMSE for training
    test_scores_mean = np.sqrt(-np.mean(test_scores, axis=1))    # RMSE for validation

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training RMSE')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation RMSE')
    plt.legend()
    plt.xlabel('Training examples')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.savefig('learning_curve.png')
    plt.close()

    # Save data to CSV
    pd.DataFrame({
        'Training_Examples': train_sizes,
        'Training_RMSE': train_scores_mean,
        'Cross_Validation_RMSE': test_scores_mean
    }).to_csv('learning_curve.csv', index=False)


def train_ensemble_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    lgb_model = LGBMRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, tree_method='hist')

    ensemble = VotingRegressor([
        ('rf', rf_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ])

    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Ensemble Model - Test set RMSE: {rmse:.4f}")
    print(f"Ensemble Model - Test set Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Ensemble Model - Test set R² Score: {r2:.4f}")

    return ensemble, X_test, y_test

    
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.savefig("feature_importance.png")
    plt.close()
    
    # Save data to CSV
    pd.DataFrame({
        'Feature': np.array(feature_names)[sorted_idx],
        'Importance': importance[sorted_idx]
    }).to_csv('feature_importance.csv', index=False)

def main():
    print_memory_usage()
    cif_file_path = "/home/mejiadongs/missions/ML/cgcnn-master_2/data/sample-regression/dielectricity/*.cif"
    y_data_path = "/home/mejiadongs/missions/ML/cgcnn-master_2/data/sample-regression/dielectricity/id_prop.csv"
    homo_lumo_path = "/home/mejiadongs/missions/ML/cgcnn-master_2/data/sample-regression/dielectricity/id_prop_humo-lumo.csv"
    
    structures, combined_data = load_data(cif_file_path, y_data_path, homo_lumo_path)
    print(f"Number of structures: {len(structures)}")
    print(f"Shape of combined_data: {combined_data.shape}")
    print_memory_usage()
    
    X = extract_features(structures, combined_data)
    y_values = combined_data.iloc[:, 0].values  # Assuming the first column is the target variable
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y_values: {y_values.shape}")
    print_memory_usage()

    # Ensure X and y_values have consistent lengths
    min_length = min(len(X), len(y_values))
    X = X.iloc[:min_length]
    y_values = y_values[:min_length]

    print(f"After adjustment - Shape of X: {X.shape}")
    print(f"After adjustment - Shape of y_values: {y_values.shape}")

    # Handle missing values
    X = X.fillna(X.mean())  # Fill NaNs with column means

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gc.collect()  # Force garbage collection
    print_memory_usage()

    # Train and evaluate XGBoost model
    xgb_model, X_test_scaled, y_test, y_pred, grid_search = train_xgboost_model(X_scaled, y_values)
    print_memory_usage()
    
    evaluate_model(xgb_model, X_test_scaled, y_test)
    plot_feature_importance(xgb_model, X.columns)
    plot_learning_curve(xgb_model, X_scaled, y_values)

    # Plot correlation matrix
    plot_correlation_matrix(X_scaled, X.columns.tolist())

    # Plot partial dependence plots for selected features
    # Prepare X_test_df with feature names
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    selected_features = ['avg_electronegativity', 'dipole_moment']  # Replace with your features of interest
    # Check that selected features exist in X_test_df
    missing_features = [feat for feat in selected_features if feat not in X_test_df.columns]
    if missing_features:
        print(f"Error: The following selected features are not in the dataset: {missing_features}")
    else:
        plot_partial_dependence(xgb_model, X_test_df, selected_features)

    # Plot SHAP summary
    # Prepare a DataFrame for SHAP with feature names
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    plot_shap_summary(xgb_model, X_test_df, X.columns)

    # Plot hyperparameter performance
    plot_hyperparameter_performance(grid_search)

    # Plot residuals
    plot_residuals(y_test, y_pred)

    # Plot error vs feature
    plot_error_vs_feature(y_test, y_pred, X_test_scaled, X.columns.tolist(), 'dipole_moment')  # Change feature_name as needed

    # Plot prediction distribution
    plot_prediction_distribution(y_test, y_pred)

    print_memory_usage()

if __name__ == "__main__":
    main()
