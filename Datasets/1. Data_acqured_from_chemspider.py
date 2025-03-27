import requests
import time
import json
import csv
from rdkit import Chem
from rdkit.Chem import AllChem

# API key and base URL
API_KEY = "XXXXXXX"  # Please replace with your actual API key
BASE_URL = "https://api.rsc.org/compounds/v1"

# Set request headers
headers = {
    "apikey": API_KEY,
    "Content-Type": "application/json"
}

def element_filter_search(include_elements, exclude_elements=None, complexity="any", isotopic="any", order_by="recordId", order_direction="ascending"):
    url = f"{BASE_URL}/filter/element"
    
    data = {
        "includeElements": include_elements,
        "excludeElements": exclude_elements or [],
        "options": {
            "includeAll": True,  # Critical parameter to ensure that returned compounds contain all specified elements
            "complexity": complexity,
            "isotopic": isotopic
        },
        "orderBy": order_by,
        "orderDirection": order_direction
    }
    
    print(f"Sending request to {url}, with data:\n{json.dumps(data, indent=2)}")
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    query_id = response.json().get("queryId")
    if not query_id:
        raise Exception("Failed to obtain queryId from the response")
    
    return query_id

def check_query_status(query_id):
    url = f"{BASE_URL}/filter/{query_id}/status"
    
    while True:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        status_data = response.json()
        print(f"Query status: {status_data}")
        
        status = status_data.get("status")
        if status == "Complete":
            return status_data.get("count", 0)
        elif status in ["Suspended", "Failed", "Not Found"]:
            raise Exception(f"Query failed with status: {status}")
        
        print("Query in progress, waiting 5 seconds before retrying...")
        time.sleep(5)  # Wait 5 seconds before checking again

def get_query_results(query_id, start=0, count=100):
    url = f"{BASE_URL}/filter/{query_id}/results"
    params = {"start": start, "count": count}
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    results = response.json()
    record_ids = results.get("results", [])
    print(f"Retrieved {len(record_ids)} record IDs")
    return record_ids

def get_compound_details(record_ids):
    url = f"{BASE_URL}/records/batch"
    data = {
        "recordIds": record_ids,
        "fields": ["SMILES", "Formula", "AverageMass", "MolecularWeight", "MonoisotopicMass", "NominalMass", "CommonName"]
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    details = response.json()
    compounds = details.get("records", [])
    print(f"Retrieved details for {len(compounds)} compounds")
    return compounds

def generate_3d_structure(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to generate molecule object from SMILES: {smiles}")
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        return mol_block
    except Exception as e:
        print(f"Error generating 3D structure: {e}")
        return None

def save_to_json(compounds, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(compounds, f, ensure_ascii=False, indent=2)
    print(f"Results saved to JSON file: {file_name}")

def save_to_csv(compounds, file_name):
    with open(file_name, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["ID", "CommonName", "SMILES", "Formula", "AverageMass", "MolecularWeight", "MonoisotopicMass", "NominalMass"])
        # Write data rows
        for compound in compounds:
            writer.writerow([
                compound.get("id", ""),
                compound.get("commonName", ""),
                compound.get("smiles", ""),
                compound.get("formula", ""),
                compound.get("averageMass", ""),
                compound.get("molecularWeight", ""),
                compound.get("monoisotopicMass", ""),
                compound.get("nominalMass", "")
            ])
    print(f"Results saved to CSV file: {file_name}")

def main():
    include_elements = ["C", "H", "O", "N", "F"]  # Elements to include
    exclude_elements = ["S", "P", "Br"]  # Elements to exclude
    max_records = 1000  # Maximum number of records to retrieve
    
    try:
        # Start element filter search
        query_id = element_filter_search(include_elements, exclude_elements)
        print(f"Obtained query ID: {query_id}")
        
        # Check query status and get total record count
        total_count = check_query_status(query_id)
        print(f"Total records found: {total_count}")
        
        if total_count == 0:
            print("No compounds matching the criteria were found")
            return
        
        # Set file names
        json_file_name = "chemspider_results.json"
        csv_file_name = "chemspider_summary.csv"
        
        processed_count = 0
        batch_size = 100
        all_compounds = []
        
        while processed_count < min(max_records, total_count):
            current_batch_size = min(batch_size, max_records - processed_count)
            record_ids = get_query_results(query_id, start=processed_count, count=current_batch_size)
            
            if not record_ids:
                print("No more record IDs retrieved, stopping the retrieval process")
                break
            
            compounds = get_compound_details(record_ids)
            
            for compound in compounds:
                smiles = compound.get("smiles")
                if smiles:
                    mol3d = generate_3d_structure(smiles)
                    if mol3d:
                        compound["mol3D"] = mol3d
                    else:
                        compound["mol3D"] = None
                else:
                    compound["mol3D"] = None
            
            all_compounds.extend(compounds)
            processed_count += len(record_ids)
            print(f"Processed {processed_count}/{min(max_records, total_count)} records")
        
        # Save results
        save_to_json(all_compounds, json_file_name)
        save_to_csv(all_compounds, csv_file_name)
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    main()
