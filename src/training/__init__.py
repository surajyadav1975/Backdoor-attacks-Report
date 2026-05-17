"""Training scripts for Phase 1 (infected baseline) and Phase 2 (secure recovery)."""

from .phase1_infected_model import train_infected_model
from .phase2_secure_recovery import train_secure_model

__all__ = ["train_infected_model", "train_secure_model"]
