class InvalidOwnerIdError(Exception):
    """Raised when a owner id is not valid."""

    def __init__(self, message="Owner ID must be a string"):
        self.message = message
        super().__init__(self.message)


class InvalidDocumentIdError(Exception):
    """Raised when a document id is not valid."""

    def __init__(self, message="Document ID must be a string"):
        self.message = message
        super().__init__(self.message)


class InvalidConversationIdError(Exception):
    """Raised when a conversation id is not valid."""

    def __init__(self, message="Conversation ID must be a string"):
        self.message = message
        super().__init__(self.message)
