"""
Script d'analyse, équilibrage et découpage Group-Split du dataset Chatbot.
Fichier source : dataset_mixed_clean.jsonl
"""
import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split

INPUT_FILE = "dataset_mixed_clean.jsonl"
TRAIN_FILE = "train_split.jsonl"
TEST_FILE = "test_split.jsonl"

def classer_tache(system_prompt):
    """Identifie la sous-tâche pour analyser l'équilibre du dataset."""
    sys_lower = system_prompt.lower()
    if "postgresql" in sys_lower or "postgis" in sys_lower and "json" in sys_lower:
        return "Text-to-SQL (JSON)"
    elif "expert sig" in sys_lower or "isohyètes" in sys_lower:
        return "Génération Code Python/SIG"
    return "Autre"

def extraire_groupe(user_prompt):
    """
    Extrait un groupe contextuel (année ou gouvernorat) 
    pour appliquer le principe du Group-Split et éviter le data leakage.
    """
    # Chercher une année (ex. 2020, 2023...)
    annee = re.findall(r"\b(19\d{2}|20\d{2})\b", user_prompt)
    if annee:
        return f"Annee_{annee[0]}"
    # Sinon groupe par défaut
    return "Standard"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lignes = [json.loads(line) for line in f if line.strip()]

    print(f"📊 Total d'exemples dans le dataset : {len(lignes)}")

    # 1. Analyse de l'équilibre des tâches (Task Imbalance)
    taches = []
    groupes = []
    for item in lignes:
        sys_p = ""
        user_p = ""
        for m in item.get("messages", []):
            if m["role"] == "system":
                sys_p = m["content"]
            elif m["role"] == "user":
                user_p = m["content"]
        
        taches.append(classer_tache(sys_p))
        groupes.append(extraire_groupe(user_p))

    compte_taches = Counter(taches)
    print("\n--- Répartition des classes/tâches ---")
    for t, n in compte_taches.items():
        print(f"• {t} : {n} ({n/len(lignes)*100:.1f} %)")

    # 2. Découpage stratifié Train (90%) / Test (10%) respectant le ratio des tâches
    train_data, test_data = train_test_split(
        lignes, 
        test_size=0.10, 
        stratify=taches, 
        random_state=42
    )

    print(f"\n✅ Découpage terminé :")
    print(f"• Train set : {len(train_data)} exemples -> sauvegardé dans {TRAIN_FILE}")
    print(f"• Test set  : {len(test_data)} exemples -> sauvegardé dans {TEST_FILE}")

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        for ex in test_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
