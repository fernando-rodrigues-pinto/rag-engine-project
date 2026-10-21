"""Parser sintático de Markdown baseado em tokens do markdown-it-py."""

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel, ConfigDict, Field

_parser: MarkdownIt = MarkdownIt("gfm-like")

_HEADING_LEVELS: dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}


class ParsedSection(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str
    headers: list[str] = Field(default_factory=list)
    is_code: bool = False


def _heading_level(tag: str) -> int:
    return _HEADING_LEVELS.get(tag, 0)


def _inline_text(token: Token) -> str:
    children = token.children
    if not children:
        return " ".join(token.content.split())
    parts: list[str] = []
    for child in children:
        child_type = child.type
        if child_type in ("text", "text_special", "code_inline"):
            parts.append(child.content)
        elif child_type in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif child_type == "image":
            alt_children = child.children
            if alt_children:
                alt_parts: list[str] = []
                for alt in alt_children:
                    if alt.type in ("text", "text_special", "code_inline"):
                        alt_parts.append(alt.content)
                alt_text = "".join(alt_parts)
                if alt_text.strip() != "":
                    parts.append(alt_text)
                elif child.content.strip() != "":
                    parts.append(child.content)
            elif child.content.strip() != "":
                parts.append(child.content)
    raw_text = "".join(parts)
    return " ".join(raw_text.split())


def _escape_table_cell(cell: str) -> str:
    return cell.replace("|", "\\|")


def _build_table_text(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    has_content = False
    for row in rows:
        for cell in row:
            if cell.strip() != "":
                has_content = True
                break
        if has_content:
            break
    if not has_content:
        return ""
    width = max(len(row) for row in rows)
    normalized: list[list[str]] = []
    for row in rows:
        padded = list(row) + [""] * (width - len(row))
        normalized.append([_escape_table_cell(cell) for cell in padded])
    lines: list[str] = []
    lines.append("| " + " | ".join(normalized[0]) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for body_row in normalized[1:]:
        lines.append("| " + " | ".join(body_row) + " |")
    return "\n".join(lines)


def parse_markdown(content: str) -> list[ParsedSection]:
    header_stack: list[tuple[int, str]] = []
    sections: list[ParsedSection] = []
    if content.strip() == "":
        return sections
    tokens: list[Token] = _parser.parse(content)
    total = len(tokens)
    index = 0
    while index < total:
        token = tokens[index]
        if token.type == "heading_open":
            level = _heading_level(token.tag)
            heading_text = ""
            if index + 1 < total and tokens[index + 1].type == "inline":
                heading_text = _inline_text(tokens[index + 1])
            if level >= 1:
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                if heading_text != "":
                    header_stack.append((level, heading_text))
            index += 1
            if index < total and tokens[index].type == "inline":
                index += 1
            if index < total and tokens[index].type == "heading_close":
                index += 1
            continue
        if token.type in ("fence", "code_block"):
            raw = token.content
            if raw.strip() != "":
                sections.append(
                    ParsedSection(
                        content=raw,
                        headers=[text for _, text in header_stack],
                        is_code=True,
                    )
                )
            index += 1
            continue
        if token.type == "table_open":
            rows: list[list[str]] = []
            current: list[str] | None = None
            cursor = index + 1
            while cursor < total and tokens[cursor].type != "table_close":
                cursor_type = tokens[cursor].type
                if cursor_type == "tr_open":
                    current = []
                elif cursor_type == "tr_close":
                    if current is not None:
                        rows.append(current)
                        current = None
                elif cursor_type == "inline":
                    if current is not None:
                        current.append(_inline_text(tokens[cursor]))
                cursor += 1
            table_text = _build_table_text(rows)
            if table_text.strip() != "":
                sections.append(
                    ParsedSection(
                        content=table_text,
                        headers=[text for _, text in header_stack],
                        is_code=False,
                    )
                )
            index = cursor + 1 if cursor < total else cursor
            continue
        if token.type == "inline":
            text = _inline_text(token)
            if text != "":
                sections.append(
                    ParsedSection(
                        content=text,
                        headers=[text for _, text in header_stack],
                        is_code=False,
                    )
                )
            index += 1
            continue
        index += 1
    return sections
