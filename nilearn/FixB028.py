import os
import re

# Chemin vers le répertoire contenant les fichiers à corriger
PROJECT_DIR = "C:/Users/hilal/nilearn/nilearn"

# Expression régulière pour détecter warnings.warn(, stacklevel=x)
WARN_REGEX = re.compile(r"(warnings\.warn\([^)]*)\)")

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        match = WARN_REGEX.search(line)
        if match:
            # Détecte si on est dans une fonction privée
            stacklevel = "2"  # Par défaut
            if re.search(r"def _", "".join(lines)):
                stacklevel = "3"
            
            # Si stacklevel est déjà présent, le remplacer. Sinon, l'ajouter.
            if "stacklevel" in match.group(1):
                new_line = re.sub(r"stacklevel=\d", f"stacklevel={stacklevel}", line)
            else:
                new_line = match.group(1) + f", stacklevel={stacklevel})"
            updated_lines.append(new_line + '\n')  # Assurez-vous d'ajouter un retour à la ligne
        else:
            updated_lines.append(line)
    
    # Écrit les modifications dans le fichier
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)

def process_project(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                process_file(os.path.join(root, file))

# Lancer le script
process_project(PROJECT_DIR)
