"""Ошибки доступа к API, при которых оценку нужно остановить и продолжить позже."""


class ServiceAccessError(RuntimeError):
    def __init__(self, service, status):
        self.service = service
        self.status = status
        super().__init__(f"{service}: HTTP {status}; проверьте ключ и доступные кредиты.")


def check_service_access(service, status, body=""):
    denied = status in {401, 402}
    if service == "Serper" and status == 403:
        denied = True
    if service == "Jina" and status == 403:
        # Отказ целевого сайта не означает, что закончились кредиты Jina.
        denied = any(word in body.lower() for word in (
            "insufficient balance", "insufficient_balance", "account balance not enough",
            "insufficient credits", "quota exceeded", "insufficient token",
        ))
    if denied:
        raise ServiceAccessError(service, status)
