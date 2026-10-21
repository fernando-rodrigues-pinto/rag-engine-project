"""Chunker hierárquico com agregação por janela e particionamento secundário."""

import uuid

from src.core.models import Chunk, ChunkMetadata
from src.ingestion.parsers import ParsedSection


def estimate_tokens(text: str) -> int:
    """Estima tokens de forma determinística como ``max(1, len(text) // 4)``."""
    return max(1, len(text) // 4)


def _breadcrumb_prefix(headers: list[str]) -> str:
    if not headers:
        return ""
    return f"[Contexto: {' > '.join(headers)}]\n\n"


def _format_chunk_content(headers: list[str], body: str) -> str:
    prefix = _breadcrumb_prefix(headers)
    if prefix == "":
        return body
    return f"{prefix}{body}"


def _ensure_breadcrumb_fits(headers: list[str], max_tokens: int) -> None:
    if estimate_tokens(_format_chunk_content(headers, "x")) > max_tokens:
        raise ValueError(
            f"max_tokens ({max_tokens}) is insufficient for section breadcrumbs "
            f"{headers!r}: minimal content already exceeds token budget"
        )


def _dedup_consecutive(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped


def _headers_compatible(a: list[str], b: list[str]) -> bool:
    """Duas linhagens são compatíveis se uma for prefixo exato da outra."""
    return b[: len(a)] == a or a[: len(b)] == b


def _merge_headers(existing: list[str], new: list[str]) -> list[str]:
    existing_dedup = _dedup_consecutive(list(existing))
    new_dedup = _dedup_consecutive(list(new))
    if not existing_dedup:
        return new_dedup
    if not new_dedup:
        return existing_dedup
    if new_dedup[: len(existing_dedup)] == existing_dedup:
        return new_dedup
    if existing_dedup[: len(new_dedup)] == new_dedup:
        return existing_dedup
    if len(new_dedup) >= len(existing_dedup):
        return new_dedup
    return existing_dedup


def _make_chunk(
    document_id: str,
    chunk_index: int,
    content: str,
    headers: list[str],
) -> Chunk:
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{chunk_index}"))
    metadata = ChunkMetadata(
        document_id=document_id,
        chunk_index=chunk_index,
        headers=list(headers),
        char_count=len(content),
        token_count=estimate_tokens(content),
    )
    return Chunk(id=chunk_id, content=content, metadata=metadata, embedding=None)


def _split_string_by_chars(
    content: str,
    headers: list[str],
    max_tokens: int,
) -> list[str]:
    _ensure_breadcrumb_fits(headers, max_tokens)
    if content.strip() == "":
        return []
    prefix_len = len(_breadcrumb_prefix(headers))
    allowed = max_tokens * 4 + 3 - prefix_len
    pieces: list[str] = []
    start = 0
    total = len(content)
    while start < total:
        end = start + allowed
        if end > total:
            end = total
        piece = content[start:end]
        formatted = _format_chunk_content(headers, piece)
        while len(piece) > 1 and estimate_tokens(formatted) > max_tokens:
            piece = piece[:-1]
            formatted = _format_chunk_content(headers, piece)
        if piece.strip() == "":
            # Fragmento apenas com espacos: descarta para nunca emitir chunk vazio.
            # Avanca o cursor sem anexar, preservando o limite de tokens.
            start += len(piece)
            if len(piece) == 0:
                break
            continue
        pieces.append(piece)
        start += len(piece)
    return [p for p in pieces if p.strip() != ""]


def _split_text(
    text: str,
    headers: list[str],
    max_tokens: int,
) -> list[str]:
    _ensure_breadcrumb_fits(headers, max_tokens)
    # Trim apenas nas bordas externas de toda a secao, preservando indentacao interna.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    trimmed = normalized.strip("\n")
    if trimmed.strip() == "":
        return []

    def _fits(body: str) -> bool:
        return estimate_tokens(_format_chunk_content(headers, body)) <= max_tokens

    if estimate_tokens(_format_chunk_content(headers, trimmed)) <= max_tokens:
        return [trimmed]

    # Particiona por paragrafos ("\n\n"), descartando apenas paragrafos vazios.
    # Preserva os espacos a esquerda de cada paragrafo (apenas "\n" extras removidos).
    raw_paragraphs = trimmed.split("\n\n")
    paragraphs: list[str] = []
    for raw in raw_paragraphs:
        if raw.strip() == "":
            continue
        para = raw.strip("\n")
        if para.strip() == "":
            continue
        paragraphs.append(para)
    if not paragraphs:
        return []
    fragments: list[str] = []
    buffer = ""
    for para in paragraphs:
        if _fits(para):
            candidate = para if buffer == "" else buffer + "\n\n" + para
            if _fits(candidate):
                buffer = candidate
            else:
                if buffer.strip() != "":
                    fragments.append(buffer)
                buffer = para
            continue
        if buffer.strip() != "":
            fragments.append(buffer)
            buffer = ""
        # Paragrafo oversized: particiona por linhas preservando indentacao.
        # Usa splitlines(keepends=True) e remove apenas "\r"/"\n", mantendo espacos.
        raw_with_ends = para.splitlines(keepends=True)
        split_lines: list[str] = []
        for chunk_line in raw_with_ends:
            line_text = chunk_line.rstrip("\r\n")
            split_lines.append(line_text)
        # Caso o paragrafo nao termine com quebra, splitlines ja cobre tudo.
        # Se o paragrafo continha "\n", split_lines reflete as linhas originais.
        line_buffer = ""
        for raw_line in split_lines:
            if raw_line.strip() == "":
                # Linha em branco ou apenas com espacos: anexa ao buffer adjacente
                # ou descarta se isolada nas pontas.
                if line_buffer != "":
                    candidate_ws = line_buffer + "\n" + raw_line
                    if _fits(candidate_ws):
                        line_buffer = candidate_ws
                    else:
                        if line_buffer.strip() != "":
                            fragments.append(line_buffer)
                        line_buffer = ""
                        # Descarta a linha em branco isolada (ponta de novo buffer).
                else:
                    # Ponta inicial: descarta para nao emitir fragmento vazio.
                    continue
                continue
            # Preserva a indentacao: usa a linha intacta, sem strip sem argumentos.
            line = raw_line
            if _fits(line):
                line_candidate = line if line_buffer == "" else line_buffer + "\n" + line
                if _fits(line_candidate):
                    line_buffer = line_candidate
                else:
                    if line_buffer.strip() != "":
                        fragments.append(line_buffer)
                    line_buffer = line
                continue
            if line_buffer.strip() != "":
                fragments.append(line_buffer)
                line_buffer = ""
            # Linha longa: particiona por palavras preservando indentacao inicial.
            indent_len = 0
            for ch in line:
                if ch == " " or ch == "\t":
                    indent_len += 1
                else:
                    break
            indent = line[:indent_len]
            stripped_line = line.strip(" \t\n\r")
            if stripped_line == "":
                continue
            words = stripped_line.split()
            word_buffer = ""
            is_first = True
            for word in words:
                if is_first and word_buffer == "" and indent != "":
                    single = indent + word
                    if not _fits(single) and _fits(word):
                        single = word
                else:
                    single = word
                if _fits(single):
                    if word_buffer == "":
                        word_candidate = single
                    else:
                        word_candidate = word_buffer + " " + word
                    if _fits(word_candidate):
                        word_buffer = word_candidate
                    else:
                        if word_buffer.strip() != "":
                            fragments.append(word_buffer)
                        is_first = False
                        word_buffer = word
                    continue
                if word_buffer.strip() != "":
                    fragments.append(word_buffer)
                    word_buffer = ""
                is_first_after_split = is_first
                char_pieces = _split_string_by_chars(word, headers, max_tokens)
                if is_first_after_split and indent != "" and char_pieces:
                    first_piece = indent + char_pieces[0]
                    if _fits(first_piece):
                        char_pieces[0] = first_piece
                for piece in char_pieces[:-1] if char_pieces else []:
                    if piece.strip() != "":
                        fragments.append(piece)
                if char_pieces:
                    last_piece = char_pieces[-1]
                    word_buffer = last_piece if last_piece.strip() != "" else ""
                else:
                    word_buffer = ""
                is_first = False
            if word_buffer.strip() != "":
                line_buffer = word_buffer
        if line_buffer.strip() != "":
            fragments.append(line_buffer)
    if buffer.strip() != "":
        fragments.append(buffer)
    return [frag for frag in fragments if frag.strip() != ""]


def _split_code(
    text: str,
    headers: list[str],
    max_tokens: int,
) -> list[str]:
    _ensure_breadcrumb_fits(headers, max_tokens)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    trimmed = normalized.strip("\n")
    if trimmed.strip() == "":
        return []
    if estimate_tokens(_format_chunk_content(headers, trimmed)) <= max_tokens:
        return [trimmed]

    def _fits(body: str) -> bool:
        return estimate_tokens(_format_chunk_content(headers, body)) <= max_tokens

    all_lines = trimmed.split("\n")
    # Descarta linhas em branco isoladas nas bordas externas.
    start_idx = 0
    while start_idx < len(all_lines) and all_lines[start_idx].strip() == "":
        start_idx += 1
    end_idx = len(all_lines)
    while end_idx > start_idx and all_lines[end_idx - 1].strip() == "":
        end_idx -= 1
    lines = all_lines[start_idx:end_idx]
    if not lines:
        return []
    fragments: list[str] = []
    buf_lines: list[str] = []
    for line in lines:
        if line.strip() == "":
            # Linha em branco ou so com espacos: anexa ao buffer adjacente
            # ou descarta se isolada nas pontas.
            if not buf_lines:
                buf_lines.append(line)
            else:
                candidate = "\n".join(buf_lines + [line])
                if _fits(candidate):
                    buf_lines.append(line)
                else:
                    buf_text = "\n".join(buf_lines)
                    if buf_text.strip() != "":
                        fragments.append(buf_text)
                    buf_lines = [line]
            continue
        if _fits(line):
            if not buf_lines:
                buf_lines.append(line)
            else:
                candidate = "\n".join(buf_lines + [line])
                if _fits(candidate):
                    buf_lines.append(line)
                else:
                    buf_text = "\n".join(buf_lines)
                    if buf_text.strip() != "":
                        fragments.append(buf_text)
                    buf_lines = [line]
            continue
        if buf_lines:
            buf_text_flush = "\n".join(buf_lines)
            if buf_text_flush.strip() != "":
                fragments.append(buf_text_flush)
            buf_lines = []
        pieces = _split_string_by_chars(line, headers, max_tokens)
        for piece in pieces[:-1] if pieces else []:
            if piece.strip() != "":
                fragments.append(piece)
        if pieces:
            last_piece = pieces[-1]
            if last_piece.strip() != "":
                buf_lines = [last_piece]
            else:
                buf_lines = []
    if buf_lines:
        buf_text_final = "\n".join(buf_lines)
        if buf_text_final.strip() != "":
            fragments.append(buf_text_final)
    return [frag for frag in fragments if frag.strip() != ""]


def chunk_sections(
    sections: list[ParsedSection],
    document_id: str,
    max_tokens: int = 384,
) -> list[Chunk]:
    """Agrupa seções consecutivas respeitando ``max_tokens`` com breadcrumbs."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    pending_texts: list[str] = []
    pending_headers: list[str] = []
    chunks: list[Chunk] = []
    chunk_index = 0

    filtered: list[tuple[str, list[str], bool]] = []
    for section in sections:
        content = section.content.replace("\r\n", "\n").replace("\r", "\n")
        headers = _dedup_consecutive(section.headers)
        if not content.strip():
            continue
        text = content.strip("\n")
        if not text.strip():
            continue
        _ensure_breadcrumb_fits(list(headers), max_tokens)
        filtered.append((text, list(headers), section.is_code))

    if not filtered:
        return []

    for text, headers, is_code in filtered:
        single_content = _format_chunk_content(headers, text)
        if estimate_tokens(single_content) > max_tokens:
            if pending_texts:
                body = "\n\n".join(pending_texts)
                content_out = _format_chunk_content(pending_headers, body)
                chunks.append(_make_chunk(document_id, chunk_index, content_out, pending_headers))
                chunk_index += 1
                pending_texts = []
                pending_headers = []
            if is_code:
                parts = _split_code(text, headers, max_tokens)
            else:
                parts = _split_text(text, headers, max_tokens)
            for part in parts:
                if part.strip() == "":
                    continue
                part_content = _format_chunk_content(headers, part)
                chunks.append(_make_chunk(document_id, chunk_index, part_content, headers))
                chunk_index += 1
            continue
        if not pending_texts:
            candidate_headers = list(headers)
            candidate_body = text
        else:
            if not _headers_compatible(pending_headers, headers):
                body = "\n\n".join(pending_texts)
                content_out = _format_chunk_content(pending_headers, body)
                chunks.append(_make_chunk(document_id, chunk_index, content_out, pending_headers))
                chunk_index += 1
                pending_headers = list(headers)
                pending_texts = [text]
                continue
            candidate_headers = _merge_headers(pending_headers, headers)
            if estimate_tokens(_format_chunk_content(candidate_headers, "x")) > max_tokens:
                body = "\n\n".join(pending_texts)
                content_out = _format_chunk_content(pending_headers, body)
                chunks.append(_make_chunk(document_id, chunk_index, content_out, pending_headers))
                chunk_index += 1
                pending_headers = list(headers)
                pending_texts = [text]
                continue
            candidate_body = "\n\n".join(pending_texts + [text])
        candidate_content = _format_chunk_content(candidate_headers, candidate_body)
        if estimate_tokens(candidate_content) <= max_tokens:
            pending_headers = candidate_headers
            pending_texts.append(text)
        else:
            body = "\n\n".join(pending_texts)
            content_out = _format_chunk_content(pending_headers, body)
            chunks.append(_make_chunk(document_id, chunk_index, content_out, pending_headers))
            chunk_index += 1
            pending_headers = list(headers)
            pending_texts = [text]

    if pending_texts:
        body = "\n\n".join(pending_texts)
        content_out = _format_chunk_content(pending_headers, body)
        chunks.append(_make_chunk(document_id, chunk_index, content_out, pending_headers))

    return chunks


class HierarchicalChunker:
    """Agrupa ``ParsedSection`` em ``Chunk`` respeitando janela de tokens."""

    def __init__(self, max_tokens: int = 384) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens: int = max_tokens

    def chunk(self, sections: list[ParsedSection], document_id: str) -> list[Chunk]:
        """Divide seções em chunks determinísticos com breadcrumbs e UUIDv5."""
        return chunk_sections(sections, document_id, self.max_tokens)
