from .in_memory import InMemoryPluginRepository
from .postgres import PostgresPluginRepository

__all__ = ["InMemoryPluginRepository", "PostgresPluginRepository"]
