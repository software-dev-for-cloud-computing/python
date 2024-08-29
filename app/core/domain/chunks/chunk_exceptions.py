class InvalidContentError(Exception):
    """Raised when the content of the chunk is invalid."""

    def __init__(self, message="Content must be a string"):
        self.message = message
        super().__init__(self.message)


class InvalidPageNumberError(Exception):
    """Raised when the page number is invalid."""

    def __init__(self, message="Page number must be a positive integer"):
        self.message = message
        super().__init__(self.message)


class ChunkNotFoundError(Exception):
    """Raised when a chunk is not found."""

    def __init__(self, message="Document not found"):
        self.message = message
        super().__init__(self.message)


class ChunkCreationError(Exception):
    """Raised when there is an error creating a chunk."""

    def __init__(self, message="Error creating chunk"):
        self.message = message
        super().__init__(self.message)
