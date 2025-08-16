Installer UV :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Vérifier l'installation :

```bash
uv --version
```

Créer le projet Kedro :

```bash
uvx --python 3.12 run kedro new --name edf_forecasting
```

Se placer dans le dossier du projet :

```bash
cd edf-forecasting
```

Supprimer l’ancien environnement si besoin :

```bash
rm -rf .venv
```

Créer un nouvel environnement :

```bash
uv venv
```

Activer l’environnement :

```bash
source .venv/bin/activate
```

Installer les dépendances :

```bash
uv pip install -r requirements.txt
```

Ajouter les dépendances au pyproject.toml :

```bash
uv add --requirements requirements.txt --active
```

Ajouter une dépendance manuellement sans toucher au fichier `requirements.txt` :

```bash
uv add package
```

Vérifier si Kedro est installé :

```bash
kedro --help
```
## Intégration de mlflow
**Ajoute les deps au projet (et lock)**
```bash
uv add mlflow "kedro-mlflow~=0.12.2"   # car ma version de Kedro est 0.12.x
```

