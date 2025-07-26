# **IMPORTANT**: activer l'environnement avant d'utiliser kedro
# Initialisation d'une pipeline :
```bash 
kedro pipeline create train_xgboost_day
```
 kedro run --help
 pip install kedro-datasets[pandas] ou uv pip install "kedro-datasets[pandas]"
pour l'ajouter au proje pyproject.toml : uv add "kedro-datasets[pandas]"

uv add "kedro-datasets[json]"
uv add "kedro-datasets[text]"
 uv add "kedro-datasets[pickle]"

**ATTENTION**
Kedro **n’exécute pas un nœud** s’il ne trouve **aucune dépendance explicite** entre les nœuds (entrées/sorties). Si un composant écrit uniquement sur le disque sans output déclaré dans le pipeline, **il sera *ignoré*** par Kedro.
Pour forcer son exécution, **déclare un output factice** (ex : `scrape_status.json`) dans le catalogue, même s’il ne sert qu’à signaler l’exécution réussie. C’est indispensable pour préserver l’ordre.

Et pour visualiser une pipeline :
installer kedro-viz via uv add kedro-viz
ensuite 
```bash
kedro viz
kedro viz --pipeline prepare_data
```
Demarrer la pipeline 
`kedro run --pipeline xgboost_day_training`

