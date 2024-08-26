import asyncio
from abc import ABC, abstractmethod
from typing import List, Callable, Tuple

class EventClass(ABC):
    def __init__(self):
        self.commands = []

    @abstractmethod
    def add_cmd(self) -> None:
        pass

    def add_cmd(self, cmd: Callable[[], asyncio.Future], mode: str, priority: int = 0) -> None:
        self.commands.append((cmd, mode, priority))

    def get_cmds(self) -> List[Tuple[Callable[[], asyncio.Future], str, int]]:
        return self.commands