"""Stable public errors; native exception details never become API messages."""


class ServiceError(Exception):
    def __init__(self, code: str, status: int = 400):
        self.code = code
        self.status = status
        super().__init__(code)
