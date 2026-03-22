"""Model role adapters for multi-LLM routing."""

from models.fast_model import FastModel
from models.reasoning_model import ReasoningModel
from models.coding_model import CodingModel

__all__ = ["FastModel", "ReasoningModel", "CodingModel"]
