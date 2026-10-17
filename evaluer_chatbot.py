"""
Script d'évaluation orientée tâche (Équivalent S-match pour LLM / Chatbot).
Vérifie la validité de la sortie générée sur le jeu de test :
1. Pour les requêtes SQL : Est-ce un JSON valide avec les clés 'sql_query' et 'target_col' ?
2. Pour le code Python : Le code compile-t-il sans erreur de syntaxe ?
"""

import json
import ast

def evaluer_reponse(tache, reponse_predite):
    """
    Évaluation fonctionnelle de la réponse générée.
    Renvoie True si la tâche est valide, False sinon.
    """
    if "SQL" in tache or "JSON" in tache:
        try:
            # Nettoyer les éventuelles balises ```json ... ```
            texte = reponse_predite.strip()
            if texte.startswith("```json"):
                texte = texte[7:]
            if texte.endswith("```"):
                texte = texte[:-3]
            data = json.loads(texte.strip())
            return "sql_query" in data and "target_col" in data
        except Exception:
            return False

    elif "Python" in tache:
        try:
            # Vérification de la syntaxe de l'arbre syntaxique (AST) Python
            code = reponse_predite.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    return True

def main():
    print("📋 Évaluation S-match / Task Success Rate prête à être branchée sur les prédictions du modèle.")

if __name__ == "__main__":
    main()
