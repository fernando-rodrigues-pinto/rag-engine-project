"""Suíte de testes do chunker hierárquico."""

import uuid

import pytest
from pydantic import ValidationError

from src.ingestion import HierarchicalChunker, chunk_sections, parse_markdown
from src.ingestion import ParsedSection as IngestionParsedSection
from src.ingestion.chunker import _merge_headers, estimate_tokens
from src.ingestion.parsers import ParsedSection


def test_estimate_tokens_is_deterministic() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("x" * 40) == 10


def test_small_sections_aggregated_into_single_chunk() -> None:
    sections = [
        ParsedSection(content="Primeiro paragrafo.", headers=["Doc"]),
        ParsedSection(content="Segundo paragrafo.", headers=["Doc"]),
        ParsedSection(content="Terceiro.", headers=["Doc"]),
    ]
    chunks = chunk_sections(sections, "doc-1", max_tokens=384)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "Primeiro paragrafo." in chunk.content
    assert "Segundo paragrafo." in chunk.content
    assert "Terceiro." in chunk.content
    assert chunk.metadata.char_count == len(chunk.content)
    assert chunk.metadata.token_count == estimate_tokens(chunk.content)
    assert chunk.metadata.token_count <= 384


def test_breadcrumb_prefix_and_metadata_headers() -> None:
    sections = [ParsedSection(content="Passo um.", headers=["Guia", "Instalacao"])]
    chunks = chunk_sections(sections, "doc-bc", max_tokens=384)

    assert len(chunks) == 1
    assert chunks[0].content == "[Contexto: Guia > Instalacao]\n\nPasso um."
    assert chunks[0].metadata.headers == ["Guia", "Instalacao"]
    assert chunks[0].metadata.document_id == "doc-bc"


def test_empty_headers_have_no_breadcrumb() -> None:
    sections = [ParsedSection(content="Texto direto.", headers=[])]
    chunks = chunk_sections(sections, "doc-no-header", max_tokens=384)

    assert len(chunks) == 1
    assert chunks[0].content == "Texto direto."
    assert chunks[0].metadata.headers == []


def test_unified_headers_deduplicate_consecutive() -> None:
    sections = [
        ParsedSection(content="Texto A.", headers=["H1"]),
        ParsedSection(content="Texto B.", headers=["H1", "H2"]),
        ParsedSection(content="Texto C.", headers=["H1", "H2"]),
    ]
    chunks = chunk_sections(sections, "doc-uni", max_tokens=384)

    assert len(chunks) == 1
    assert chunks[0].content.startswith("[Contexto: H1 > H2]\n\n")
    assert chunks[0].metadata.headers == ["H1", "H2"]


def test_long_text_section_is_split_into_smaller_fragments() -> None:
    body = "\n\n".join(f"Paragrafo {i} com conteudo suficiente para preencher." for i in range(10))
    sections = [ParsedSection(content=body, headers=["Longa"], is_code=False)]
    chunks = chunk_sections(sections, "doc-long-text", max_tokens=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.token_count <= 30
        assert chunk.metadata.headers == ["Longa"]
        assert chunk.content.startswith("[Contexto: Longa]\n\n")
        assert chunk.metadata.char_count == len(chunk.content)
        assert chunk.metadata.token_count == estimate_tokens(chunk.content)


def test_long_code_section_is_split_by_lines() -> None:
    code = "\n".join(f"linha {i}: x = {i} + funcao(y, z)" for i in range(30))
    sections = [ParsedSection(content=code, headers=["Codigo"], is_code=True)]
    chunks = chunk_sections(sections, "doc-long-code", max_tokens=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.token_count <= 30
        assert chunk.metadata.headers == ["Codigo"]
        assert chunk.metadata.char_count == len(chunk.content)
        assert chunk.metadata.token_count == estimate_tokens(chunk.content)
    combined = "\n".join(chunk.content for chunk in chunks)
    assert "linha 0" in combined
    assert "linha 29" in combined


def test_code_block_isolated_when_overflow_accumulator() -> None:
    sections = [
        ParsedSection(content="x" * 24, headers=[], is_code=False),
        ParsedSection(content="y" * 24, headers=[], is_code=True),
    ]
    chunks = chunk_sections(sections, "doc-mix", max_tokens=10)

    assert len(chunks) == 2
    assert chunks[0].content == "x" * 24
    assert chunks[1].content == "y" * 24
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[1].metadata.chunk_index == 1


def test_oversized_code_after_accumulator_flushes_first() -> None:
    small = ParsedSection(content="curto", headers=["H"], is_code=False)
    big_code = "\n".join(f"codigo linha numero {i} com conteudo" for i in range(20))
    sections = [
        small,
        ParsedSection(content=big_code, headers=["H"], is_code=True),
    ]
    chunks = chunk_sections(sections, "doc-flush", max_tokens=20)

    assert len(chunks) > 1
    assert chunks[0].content == "[Contexto: H]\n\ncurto"
    for chunk in chunks[1:]:
        assert chunk.metadata.headers == ["H"]
        assert chunk.metadata.token_count <= 20


def test_determinism_and_uuidv5_format() -> None:
    sections = [
        ParsedSection(content="Alpha.", headers=["T"]),
        ParsedSection(content="Beta.", headers=["T"]),
    ]
    first = chunk_sections(sections, "doc-det", max_tokens=384)
    second = chunk_sections(sections, "doc-det", max_tokens=384)

    assert [c.id for c in first] == [c.id for c in second]
    assert [c.content for c in first] == [c.content for c in second]
    for index, chunk in enumerate(first):
        parsed = uuid.UUID(chunk.id)
        assert parsed.version == 5
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc-det:{index}"))
        assert chunk.id == expected


def test_chunk_index_is_sequential() -> None:
    sections = [
        ParsedSection(content="a" * 24, headers=[], is_code=False),
        ParsedSection(content="b" * 24, headers=[], is_code=False),
        ParsedSection(content="c" * 24, headers=[], is_code=False),
    ]
    chunks = chunk_sections(sections, "doc-seq", max_tokens=10)

    assert len(chunks) == 3
    assert [c.metadata.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.metadata.document_id for c in chunks] == ["doc-seq"] * 3


def test_empty_sections_return_empty_list() -> None:
    assert chunk_sections([], "doc-empty") == []
    assert chunk_sections([ParsedSection(content="   ", headers=[])], "doc-empty") == []
    assert chunk_sections([ParsedSection(content="\n\n   \n", headers=["H"])], "doc-empty") == []
    assert HierarchicalChunker().chunk([], "doc-empty") == []


def test_chunk_and_metadata_are_frozen() -> None:
    chunks = chunk_sections([ParsedSection(content="imutavel", headers=["H"])], "doc-frozen")
    assert len(chunks) == 1
    chunk = chunks[0]
    with pytest.raises(ValidationError):
        chunk.content = "mutado"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        chunk.metadata.chunk_index = 99  # type: ignore[misc]


def test_equivalence_chunker_and_function() -> None:
    sections = [
        ParsedSection(content="Um.", headers=["T"]),
        ParsedSection(content="Dois.", headers=["T", "S"]),
        ParsedSection(content="Tres.", headers=["T", "S"]),
    ]
    via_class = HierarchicalChunker(max_tokens=50).chunk(sections, "doc-eq")
    via_func = chunk_sections(sections, "doc-eq", max_tokens=50)

    assert via_class == via_func

    default_class = HierarchicalChunker().chunk(sections, "doc-eq2")
    default_func = chunk_sections(sections, "doc-eq2")
    assert default_class == default_func


def test_ingestion_package_exports() -> None:
    import src.ingestion as ingestion

    assert ingestion.HierarchicalChunker is HierarchicalChunker
    assert ingestion.chunk_sections is chunk_sections
    assert ingestion.ParsedSection is IngestionParsedSection
    assert ingestion.parse_markdown is parse_markdown
    assert set(ingestion.__all__) == {
        "HierarchicalChunker",
        "chunk_sections",
        "ParsedSection",
        "parse_markdown",
    }


def test_char_and_token_counts_match_content() -> None:
    sections = [ParsedSection(content="Contagem exata.", headers=["C"])]
    chunks = chunk_sections(sections, "doc-count")

    assert len(chunks) == 1
    content = chunks[0].content
    assert chunks[0].metadata.char_count == len(content)
    assert chunks[0].metadata.token_count == max(1, len(content) // 4)


def test_max_tokens_non_positive_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        HierarchicalChunker(max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        HierarchicalChunker(max_tokens=-1)
    sections = [ParsedSection(content="texto", headers=["H"])]
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        chunk_sections(sections, "doc-zero", max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        chunk_sections(sections, "doc-zero", max_tokens=-5)


def test_insufficient_max_tokens_for_breadcrumb_raises() -> None:
    long_headers = ["H" * 50]
    sections = [ParsedSection(content="x", headers=long_headers)]
    with pytest.raises(ValueError, match="breadcrumb"):
        chunk_sections(sections, "doc-bc-small", max_tokens=10)
    with pytest.raises(ValueError, match="breadcrumb"):
        HierarchicalChunker(max_tokens=10).chunk(sections, "doc-bc-small")


def test_code_block_aggregated_when_fits_with_text() -> None:
    sections = [
        ParsedSection(content="Introducao breve.", headers=["Doc"], is_code=False),
        ParsedSection(content="x = 1\ny = 2", headers=["Doc"], is_code=True),
    ]
    chunks = chunk_sections(sections, "doc-code-agg", max_tokens=384)

    assert len(chunks) == 1
    assert "Introducao breve." in chunks[0].content
    assert "x = 1" in chunks[0].content
    assert "y = 2" in chunks[0].content
    assert chunks[0].metadata.token_count <= 384


def test_code_integrity_preserves_blank_lines_and_docstring() -> None:
    code = 'def foo():\n    """Docstring.\n\n    Detalhes."""\n\n    x = 1\n\n    return x'
    sections = [ParsedSection(content=code, headers=["C"], is_code=True)]
    chunks = chunk_sections(sections, "doc-code-integrity", max_tokens=15)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.token_count <= 15
        assert chunk.metadata.headers == ["C"]
    prefix = "[Contexto: C]\n\n"
    bodies = [
        chunk.content[len(prefix) :] if chunk.content.startswith(prefix) else chunk.content
        for chunk in chunks
    ]
    recombined = "\n".join(bodies)
    assert recombined == code
    assert recombined.count("\n\n") == code.count("\n\n")
    assert "" in recombined.split("\n")
    assert '"""Docstring.' in recombined
    assert "Detalhes." in recombined


def test_text_secondary_split_preserves_single_breaks() -> None:
    text = "\n".join(f"frase {i} com algum conteudo" for i in range(10))
    sections = [ParsedSection(content=text, headers=["T"], is_code=False)]
    chunks = chunk_sections(sections, "doc-text-preserve", max_tokens=20)

    assert len(chunks) > 1
    prefix = "[Contexto: T]\n\n"
    bodies: list[str] = []
    for chunk in chunks:
        assert chunk.metadata.token_count <= 20
        assert chunk.content.startswith(prefix)
        body = chunk.content[len(prefix) :]
        assert "\n\n" not in body
        bodies.append(body)
    assert "\n".join(bodies) == text


def test_merge_headers_strictly_consecutive() -> None:
    assert _merge_headers(["A", "B"], ["B", "A"]) == ["B", "A"]
    assert _merge_headers(["A"], ["A", "B"]) == ["A", "B"]
    assert _merge_headers(["H1"], ["H1", "H2"]) == ["H1", "H2"]


def test_merge_headers_universal_dedup_ciclo3() -> None:
    assert _merge_headers(["A"], ["A", "A"]) == ["A"]
    assert _merge_headers(["A", "B"], ["B", "A"]) == ["B", "A"]


def test_merge_headers_never_emits_consecutive_duplicates() -> None:
    cases: list[tuple[list[str], list[str]]] = [
        (["A"], ["A", "A"]),
        (["A", "B"], ["B", "A"]),
        ([], ["A", "A"]),
        (["A", "A"], []),
        (["A"], ["A"]),
        (["H1"], ["H1", "H1", "H2"]),
        (["X", "Y"], ["Y", "Y", "Z"]),
    ]
    for existing, new in cases:
        merged = _merge_headers(existing, new)
        for first, second in zip(merged, merged[1:], strict=False):
            assert not (first == second)


def test_merge_headers_overlap_and_fallback_paths_dedup() -> None:
    assert _merge_headers(["Doc"], ["Doc", "Doc"]) == ["Doc"]
    assert _merge_headers(["A", "B"], ["C", "C", "D"]) == ["C", "D"]
    assert _merge_headers([], ["B", "B", "C"]) == ["B", "C"]


def test_no_empty_or_whitespace_only_chunks() -> None:
    sections = [
        ParsedSection(content="Texto valido.", headers=["H"]),
        ParsedSection(content="   ", headers=["H"]),
        ParsedSection(content="\n\n   \n", headers=["H"]),
        ParsedSection(content="Outro texto.", headers=["H"]),
    ]
    chunks = chunk_sections(sections, "doc-no-empty", max_tokens=384)
    assert len(chunks) > 0
    assert all(c.content.strip() != "" for c in chunks)
    for chunk in chunks:
        body = (
            chunk.content.split("\n\n", 1)[-1] if "[Contexto:" in chunk.content else chunk.content
        )
        assert body.strip() != ""


def test_code_with_pure_space_lines_emits_no_empty_chunks() -> None:
    code = "x = 1\n    \n\n    \ny = 2\n   \nprint(x)"
    sections = [ParsedSection(content=code, headers=["C"], is_code=True)]
    chunks = chunk_sections(sections, "doc-code-spaces", max_tokens=15)
    assert len(chunks) > 0
    assert all(c.content.strip() != "" for c in chunks)
    for chunk in chunks:
        assert chunk.content.strip() != ""
        assert chunk.metadata.token_count <= 15


def test_code_split_with_blank_space_lines_no_whitespace_fragments() -> None:
    code_lines = ["linha codigo numero %d com conteudo extra" % i for i in range(12)]
    code_lines.insert(3, "    ")
    code_lines.insert(7, "")
    code_lines.insert(9, "   ")
    code = "\n".join(code_lines)
    sections = [ParsedSection(content=code, headers=["C"], is_code=True)]
    chunks = chunk_sections(sections, "doc-code-blank", max_tokens=20)
    assert len(chunks) > 1
    assert all(c.content.strip() != "" for c in chunks)
    bodies: list[str] = []
    prefix = "[Contexto: C]\n\n"
    for chunk in chunks:
        body = chunk.content[len(prefix) :] if chunk.content.startswith(prefix) else chunk.content
        assert body.strip() != ""
        bodies.append(body)
    assert all(b.strip() != "" for b in bodies)


def test_text_indentation_preserved_nested_list() -> None:
    text = "Item 1\n    - subitem A\n    - subitem B\nItem 2 final"
    sections = [ParsedSection(content=text, headers=["T"], is_code=False)]
    chunks = chunk_sections(sections, "doc-indent", max_tokens=12)
    assert len(chunks) >= 1
    assert all(c.content.strip() != "" for c in chunks)
    combined = "\n".join(c.content for c in chunks)
    assert "    - subitem A" in combined
    assert "    - subitem B" in combined


def test_split_text_preserves_four_space_indent_when_forced_split() -> None:
    filler = " ".join(f"palavra{i}" for i in range(30))
    text = f"Introducao.\n    - item aninhado com {filler}\nConclusao."
    sections = [ParsedSection(content=text, headers=["T"], is_code=False)]
    chunks = chunk_sections(sections, "doc-indent-split", max_tokens=25)
    assert len(chunks) > 1
    assert all(c.content.strip() != "" for c in chunks)
    combined = "\n".join(c.content for c in chunks)
    assert "    - item aninhado" in combined


def test_text_whitespace_only_fragments_are_attached_or_discarded() -> None:
    text = "Primeira linha.\n    \nSegunda linha com conteudo suficiente para split. " + "x" * 200
    sections = [ParsedSection(content=text, headers=["T"], is_code=False)]
    chunks = chunk_sections(sections, "doc-ws-attach", max_tokens=30)
    assert len(chunks) > 0
    assert all(c.content.strip() != "" for c in chunks)


def test_chunk_sections_ignores_whitespace_subfragments() -> None:
    sections = [
        ParsedSection(content="curto", headers=["H"], is_code=False),
        ParsedSection(
            content="codigo linha numero 1 com conteudo\n    \ncodigo linha 2",
            headers=["H"],
            is_code=True,
        ),
    ]
    chunks = chunk_sections(sections, "doc-ignore-ws", max_tokens=20)
    assert len(chunks) > 0
    assert all(c.content.strip() != "" for c in chunks)


def test_sibling_branches_trigger_flush_and_separate_chunks() -> None:
    sections = [
        ParsedSection(content="Conteudo Linux.", headers=["Guia", "Instalação", "Linux"]),
        ParsedSection(content="Conteudo Windows.", headers=["Guia", "Instalação", "Windows"]),
    ]
    chunks = chunk_sections(sections, "doc-siblings", max_tokens=384)
    assert len(chunks) == 2
    assert chunks[0].content == "[Contexto: Guia > Instalação > Linux]\n\nConteudo Linux."
    assert chunks[1].content == "[Contexto: Guia > Instalação > Windows]\n\nConteudo Windows."
    assert chunks[0].metadata.headers == ["Guia", "Instalação", "Linux"]
    assert chunks[1].metadata.headers == ["Guia", "Instalação", "Windows"]
    assert "Windows" not in chunks[0].content
    assert "Linux" not in chunks[1].content


def test_single_section_dedup_consecutive_headers() -> None:
    sections = [ParsedSection(content="Texto isolado.", headers=["A", "A", "B"])]
    chunks = chunk_sections(sections, "doc-dedup-single", max_tokens=384)
    assert len(chunks) == 1
    assert "[Contexto: A > B]" in chunks[0].content
    assert chunks[0].metadata.headers == ["A", "B"]


def test_first_line_indentation_preserved() -> None:
    sections = [ParsedSection(content="    - subitem aninhado", headers=["T"])]
    chunks = chunk_sections(sections, "doc-indent-first", max_tokens=384)
    assert len(chunks) == 1
    prefix = "[Contexto: T]\n\n"
    assert chunks[0].content.startswith(prefix)
    body = chunks[0].content[len(prefix) :]
    assert body == "    - subitem aninhado"
    assert body.startswith("    ")
    assert not body.startswith("     ")


def test_crlf_normalization_preserves_clean_lf() -> None:
    sections = [ParsedSection(content="linha1\r\nlinha2\r\nlinha3", headers=["T"])]
    chunks = chunk_sections(sections, "doc-crlf", max_tokens=384)
    assert len(chunks) == 1
    assert "\r" not in chunks[0].content
    assert "linha1\nlinha2\nlinha3" in chunks[0].content


def test_end_to_end_parse_to_chunk_pipeline() -> None:
    md = (
        "# Guia\n\nIntro geral.\n\n## Instalação\n\nPassos gerais.\n\n"
        "### Linux\n\nComandos linux aqui.\n\n### Windows\n\nComandos windows aqui.\n"
    )
    sections = parse_markdown(md)
    assert len(sections) >= 4
    chunker = HierarchicalChunker(max_tokens=384)
    chunks = chunker.chunk(sections, "doc-e2e")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "\r" not in chunk.content
        assert chunk.metadata.token_count <= 384
        assert chunk.metadata.token_count == max(1, len(chunk.content) // 4)
        assert chunk.metadata.char_count == len(chunk.content)
    linux_chunks = [c for c in chunks if "linux" in c.content.lower()]
    windows_chunks = [c for c in chunks if "windows" in c.content.lower()]
    assert len(linux_chunks) >= 1
    assert len(windows_chunks) >= 1
    for chunk in chunks:
        assert not ("Linux" in chunk.metadata.headers and "Windows" in chunk.metadata.headers)
