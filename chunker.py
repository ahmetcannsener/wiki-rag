import re
from dataclasses import dataclass

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    entity_title: str
    entity_type: str
    chunk_index: int
    source_url: str


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def _words(text: str) -> list[str]:
    return text.split()


def chunk_text(
    text: str,
    entity_title: str,
    entity_type: str,
    source_url: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    sentences = _split_sentences(text)

    chunks: list[Chunk] = []
    current_words: list[str] = []
    chunk_index = 0

    for sentence in sentences:
        sentence_words = _words(sentence)

        if current_words and len(current_words) + len(sentence_words) > chunk_size:
            chunk_text_str = " ".join(current_words)
            chunks.append(
                Chunk(
                    text=chunk_text_str,
                    entity_title=entity_title,
                    entity_type=entity_type,
                    chunk_index=chunk_index,
                    source_url=source_url,
                )
            )
            chunk_index += 1
            # carry over tail words for overlap
            current_words = current_words[-overlap:] if overlap > 0 else []

        current_words.extend(sentence_words)

    # flush remaining words
    if current_words:
        chunks.append(
            Chunk(
                text=" ".join(current_words),
                entity_title=entity_title,
                entity_type=entity_type,
                chunk_index=chunk_index,
                source_url=source_url,
            )
        )

    return chunks
