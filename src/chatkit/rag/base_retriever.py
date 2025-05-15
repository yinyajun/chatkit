from abc import ABC, abstractmethod
from typing import Protocol, Generator


class Document(Protocol):
    def id(self) -> str:
        ...

    @property
    @abstractmethod
    def value(self):
        ...


class Store(ABC):
    @abstractmethod
    def get(self, doc_id: str) -> Document:
        ...

    @abstractmethod
    def add(self, docs: list[Document]) -> int:
        ...

    @abstractmethod
    def update(self, doc_id: str, doc: Document):
        ...

    @abstractmethod
    def delete(self, doc_id: str) -> Document:
        ...

    @abstractmethod
    def traverse(self) -> Generator[Document, None]:
        ...


class Retriever(Store):
    @abstractmethod
    def retrieve(self, query: str, k: int = 1) -> list[Document]:
        ...
