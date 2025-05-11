import optuna
from kedro.io import AbstractDataset
from typing import Any, Dict

class OptunaStudyDataset(AbstractDataset):
    def __init__(self, filepath: str, study_name: str):
        self._filepath = filepath
        self._study_name = study_name

    def _load(self) -> optuna.Study:
        storage = f"sqlite:///{self._filepath}"
        return optuna.load_study(study_name=self._study_name, storage=storage)

    def _save(self, study: optuna.Study) -> None:
        # Optuna enregistre automatiquement via la méthode optimize()
        pass

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": self._filepath, "study_name": self._study_name}
