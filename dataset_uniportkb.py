import requests
import pandas as pd
from collections import Counter
import time
from Bio import SeqIO
from io import StringIO
import os
from datetime import datetime

# Evidence code scoring system
evidence_scores = {
    'EXP': 5, 'IDA': 5, 'IPI': 4, 'IMP': 4, 'IGI': 4, 'IEP': 4, 'IC': 4,
    'ISS': 3, 'ISO': 3, 'ISM': 3, 'IBA': 3, 'IEA': 1
}

# Canonical labels and their keyword mapping
label_keywords_map = {
    "Mitochondrion": ["mitochondrion", "mitochondrial", "mitochondrion outer membrane", "mitochondrial membrane"],
    "Plasma Membrane": ["plasma membrane", "cell membrane", "membrane"],
    "Endoplasmic Reticulum": ["endoplasmic reticulum", "er membrane"],
    "Golgi Apparatus": ["golgi", "golgi apparatus"],
    "Lysosome": ["lysosome"],
    "Peroxisome": ["peroxisome"],
    "Ribosome": ["ribosome"],
    "Proteasome": ["proteasome"],
    "Chloroplast": ["chloroplast"],
    "Extracellular Region": ["extracellular", "secreted", "extracellular region"],
    "Nucleus": ["nucleus", "nuclear"],
    "Cytoplasm": ["cytoplasm", "cytosolic"]
}

# GO term to localization label
go_localization_map = {
    "GO:0005737": "Cytoplasm",
    "GO:0005634": "Nucleus",
    "GO:0005739": "Mitochondrion",
    "GO:0005773": "Vacuole",
    "GO:0005886": "Plasma Membrane",
    "GO:0005783": "Endoplasmic Reticulum",
    "GO:0005794": "Golgi Apparatus",
    "GO:0005764": "Lysosome",
    "GO:0005777": "Peroxisome",
    "GO:0009536": "Plastid",
    "GO:0031965": "Nuclear Membrane",
    "GO:0005840": "Ribosome",
    "GO:0009507": "Chloroplast",
    "GO:0005576": "Extracellular Region",
    "GO:0000502": "Proteasome"
}

# Proteome IDs for each group/organism
PROTEOME_IDS = {
    "Monotremata": "UP000002279",
    "Proboscidea": "UP000007646",
    "Xenarthra": "UP000030104",
    "Primates": {
        "Human": "UP000005640",
        "Chimpanzee": "UP000002277",
        "Rhesus macaque": "UP000007266"
    },
    "Lagomorpha": {
        "Rabbit": "UP000001811"  # Oryctolagus cuniculus
    },
    "Rodentia": {
        "Mouse": "UP000000589",
        "Rat": "UP000002494",
        "Squirrel": "UP000265140"
    },
    "Carnivora": {
        "Dog": "UP000002254",
        "Cat": "UP000011142",
        "Polar bear": "UP000286593"
    },
    "Chiroptera": {
        "Big brown bat": "UP000030665",
        "Egyptian fruit bat": "UP000261642"
    },
    "Cetartiodactyla": {
        "Cow": "UP000009136",
        "Pig": "UP000008227",
        "Dolphin": "UP000266516"
    },
    "Perissodactyla": {
        "Horse": "UP000002281",
        "White rhinoceros": "UP000504744"
    }
}

def normalize_label(label):
    if not isinstance(label, str):
        return "Unknown"
    label_lower = label.lower()
    for canonical, keywords in label_keywords_map.items():
        if any(kw in label_lower for kw in keywords):
            return canonical
    return "Unknown"

def fetch_go_annotations_from_json(data):
    """Extract GO annotations from already-fetched UniProt JSON"""
    go_annotations = []
    for dbref in data.get("uniProtKBCrossReferences", []):
        if dbref.get("database") == "GO":
            go_id = dbref.get("id", "")
            evidence_code = None
            for prop in dbref.get("properties", []):
                if prop.get("key") == "GoEvidenceType":
                    evidence_code = prop.get("value", "").split(":")[0]
            for prop in dbref.get("properties", []):
                if prop.get("key") == "GoTerm" and prop.get("value", "").startswith("C:"):
                    go_annotations.append((go_id, evidence_code))
    return go_annotations

def extract_localization(accession_id):
    """Determine localization from GO annotations and fallback comments"""
    url = f"https://rest.uniprot.org/uniprotkb/{accession_id}.json"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return ("Unknown", 0)
        data = response.json()

        # Parse GO-based localization
        go_annotations = fetch_go_annotations_from_json(data)
        localization_scores = Counter()

        for go_id, evidence_code in go_annotations:
            if go_id in go_localization_map and evidence_code in evidence_scores:
                label = normalize_label(go_localization_map[go_id])
                if label != "Unknown":
                    localization_scores[label] += evidence_scores[evidence_code]

        if localization_scores:
            max_score = max(localization_scores.values())
            if max_score >= 3:
                best_labels = [label for label, score in localization_scores.items() if score == max_score]
                return (", ".join(best_labels), max_score)

        # Fallback to subcellular location comments
        for comment in data.get("comments", []):
            if comment.get("type") == "SUBCELLULAR LOCATION":
                locations = [loc.get("location", {}).get("value") for loc in comment.get("subcellularLocations", [])]
                if locations:
                    return (normalize_label(locations[0]), 2)

        return ("Unknown", 0)
    except Exception:
        return ("Unknown", 0)

def get_proteome_data(proteome_id):
    """Download and parse proteome FASTA data"""
    url = f"https://rest.uniprot.org/uniprotkb/stream?query=proteome:{proteome_id}&format=fasta"
    try:
        print(f"Fetching proteome {proteome_id} from UniProt...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        if not response.text.startswith(">"):
            raise ValueError("Invalid FASTA response")
        proteins = []
        for record in SeqIO.parse(StringIO(response.text), "fasta"):
            try:
                protein_id = record.id.split("|")[1]
                sequence = str(record.seq)
                proteins.append({"Protein_ID": protein_id, "Sequence": sequence})
            except Exception:
                continue
        print(f"Fetched {len(proteins)} proteins")
        return pd.DataFrame(proteins)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def process_proteome(group_name, organism_name=None, batch_size=1000):
    """Process and annotate proteins for a given group or organism"""
    if group_name not in PROTEOME_IDS:
        raise ValueError(f"Unknown group: {group_name}")
    if isinstance(PROTEOME_IDS[group_name], dict):
        if organism_name is None:
            raise ValueError(f"Specify organism for group {group_name}")
        proteome_id = PROTEOME_IDS[group_name][organism_name]
        output_name = f"{group_name}_{organism_name.replace(' ', '_')}"
    else:
        proteome_id = PROTEOME_IDS[group_name]
        output_name = group_name

    df = get_proteome_data(proteome_id)
    if df.empty:
        print(f"No data for {output_name}")
        return pd.DataFrame()

    # os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"/content/drive/MyDrive/esm/phylo/results/{output_name}_{timestamp}.csv"
    results = []
    start = time.time()

    for i, row in df.iterrows():
        try:
            loc, score = extract_localization(row['Protein_ID'])
            results.append({
                "Protein_ID": row['Protein_ID'],
                "Sequence": row['Sequence'],
                "Localization": loc,
                "Confidence_Score": score
            })

            if (i + 1) % batch_size == 0 or (i + 1) == len(df):
                elapsed = time.time() - start
                est_remain = (len(df) - (i + 1)) * (elapsed / (i + 1))
                print(f"Processed {i+1}/{len(df)} | Elapsed: {elapsed/60:.1f} min | "
                      f"Remaining: {est_remain/60:.1f} min")
                pd.DataFrame(results).to_csv(csv_path, index=False)

            time.sleep(0.1)  # Respect API rate limits (adjust as needed)

        except Exception as e:
            print(f"Error on {row['Protein_ID']}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"\nDone {output_name}: {len(results)} proteins")
    print(f"Results saved at: {csv_path}")
    return results_df

# Entry point
if __name__ == "__main__":
    organisms_to_process = [
        ("Perissodactyla", "White rhinoceros"),
        ("Cetartiodactyla", "Dolphin"),
         ("Carnivora", "Polar bear"),
         ('Lagomorpha', 'Rabbit')
        #  ('Rodentia', 'Mouse'),
        #  ('Primates', 'Human'),
        #  ('Monotremata', 'Xenarthra'),
        #  ('Proboscidea', 'Xenarthra




    ]

    for group, organism in organisms_to_process:
        try:
            process_proteome(group, organism)
        except Exception as e:
            print(f"Fatal error with {group} {organism or ''}: {e}")
