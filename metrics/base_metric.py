from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @property
    @abstractmethod
    def requires_references(self) -> bool:
        """
        Whether this metric requires reference captions.
        """
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def compute_score(self, *args, **kwargs):
        pass