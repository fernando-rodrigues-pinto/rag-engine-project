"""Pacote de ingestão: parsing Markdown e chunking hierárquico."""

from src.ingestion.chunker import HierarchicalChunker, chunk_sections
from src.ingestion.parsers import ParsedSection, parse_markdown

__all__ = ["HierarchicalChunker", "chunk_sections", "ParsedSection", "parse_markdown"]
