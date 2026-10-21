"""Suíte de testes do parser sintático de Markdown."""

from src.ingestion.parsers import ParsedSection, parse_markdown


def test_deep_heading_hierarchy() -> None:
    content = "# H1\n\nTexto do H1\n\n## H2\n\nTexto do H2\n\n### H3\n\nTexto do H3\n"
    sections = parse_markdown(content)

    assert len(sections) == 3
    assert sections[0] == ParsedSection(content="Texto do H1", headers=["H1"])
    assert sections[1] == ParsedSection(content="Texto do H2", headers=["H1", "H2"])
    assert sections[2] == ParsedSection(content="Texto do H3", headers=["H1", "H2", "H3"])
    for section in sections:
        assert section.is_code is False


def test_sibling_transition_discards_previous_levels() -> None:
    content = (
        "# H1\n\nTexto H1\n\n## H2\n\nTexto H2\n\n### H3\n\nTexto H3\n\n## H2_B\n\nTexto H2_B\n"
    )
    sections = parse_markdown(content)

    assert len(sections) == 4
    assert sections[0] == ParsedSection(content="Texto H1", headers=["H1"])
    assert sections[1] == ParsedSection(content="Texto H2", headers=["H1", "H2"])
    assert sections[2] == ParsedSection(content="Texto H3", headers=["H1", "H2", "H3"])
    assert sections[3] == ParsedSection(content="Texto H2_B", headers=["H1", "H2_B"])


def test_header_hierarchy_level_jumps() -> None:
    content = "# H1\n\nTexto H1\n\n### H3\n\nTexto H3\n\n## H2\n\nTexto H2\n"
    sections = parse_markdown(content)

    assert len(sections) == 3
    assert sections[0] == ParsedSection(content="Texto H1", headers=["H1"])
    assert sections[1] == ParsedSection(content="Texto H3", headers=["H1", "H3"])
    assert sections[2] == ParsedSection(content="Texto H2", headers=["H1", "H2"])


def test_fenced_code_block_preserves_content() -> None:
    content = "# Guia\n\n```python\nprint('ola')\nfor i in range(3):\n    print(i)\n```\n"
    expected_code = "print('ola')\nfor i in range(3):\n    print(i)\n"
    sections = parse_markdown(content)

    assert len(sections) == 1
    assert sections[0] == ParsedSection(content=expected_code, headers=["Guia"], is_code=True)
    assert sections[0].is_code is True


def test_indented_code_block_is_code() -> None:
    content = "# Título\n\n    linha um\n    linha dois\n"
    expected_code = "linha um\nlinha dois\n"
    sections = parse_markdown(content)

    assert len(sections) == 1
    assert sections[0] == ParsedSection(content=expected_code, headers=["Título"], is_code=True)


def test_document_without_headers_returns_empty_headers() -> None:
    content = "Apenas um parágrafo.\n\nOutro parágrafo.\n"
    sections = parse_markdown(content)

    assert len(sections) == 2
    assert sections[0] == ParsedSection(content="Apenas um parágrafo.", headers=[])
    assert sections[1] == ParsedSection(content="Outro parágrafo.", headers=[])


def test_gfm_table_grouped_in_single_section() -> None:
    content = "# Dados\n\n| Nome | Idade |\n| --- | --- |\n| Ana | 30 |\n| Bob | 25 |\n"
    expected_table = "| Nome | Idade |\n| --- | --- |\n| Ana | 30 |\n| Bob | 25 |"
    sections = parse_markdown(content)

    assert len(sections) == 1
    assert sections[0] == ParsedSection(content=expected_table, headers=["Dados"], is_code=False)


def test_table_with_literal_pipe_escapes() -> None:
    content = "# Dados\n\n| Cmd | Desc |\n| --- | --- |\n| cat file \\| grep err | filtra |\n"
    expected_table = "| Cmd | Desc |\n| --- | --- |\n| cat file \\| grep err | filtra |"
    sections = parse_markdown(content)

    assert len(sections) == 1
    assert sections[0].headers == ["Dados"]
    assert sections[0].is_code is False
    assert sections[0].content == expected_table


def test_headers_are_not_emitted_as_sections() -> None:
    content = "# H1\n\n## H2\n\nTexto final\n"
    sections = parse_markdown(content)

    assert len(sections) == 1
    assert sections[0] == ParsedSection(content="Texto final", headers=["H1", "H2"])


def test_whitespace_only_sections_are_ignored() -> None:
    assert parse_markdown("") == []
    assert parse_markdown("   \n\n  \n") == []
    sections = parse_markdown("# H1\n\nTexto válido\n\n   \n")

    assert sections == [ParsedSection(content="Texto válido", headers=["H1"])]


def test_list_items_inherit_headers() -> None:
    content = "# Lista\n\n- item um\n- item dois\n"
    sections = parse_markdown(content)

    assert sections == [
        ParsedSection(content="item um", headers=["Lista"]),
        ParsedSection(content="item dois", headers=["Lista"]),
    ]
