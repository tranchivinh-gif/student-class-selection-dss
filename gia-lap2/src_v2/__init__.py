"""End-to-end elective subject recommendation system."""

from .config import FEATURES, TARGETS
from .model import SubjectRankingModel

__all__ = ["FEATURES", "TARGETS", "SubjectRankingModel"]
