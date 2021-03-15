class ModelInferenceError(ValueError):
    """Raised when the model can't evaluate some input"""
    pass


class TripleExtractionError(Exception):
    """Raised when input cannot be parsed."""
    pass
