"""Generate the ER diagram as draw.io XML, from the live database.

Written rather than drawn, and read from `information_schema` rather than from
a model file, because a diagram that is maintained by hand is a diagram that
is wrong. Re-run it after a migration and the picture is correct again:

    python scripts/er_drawio.py            # writes docs/diagrams/er.drawio
    drawio -x -f png --scale 2.5 -o er.png docs/diagrams/er.drawio

Tables are laid out in three columns by lifecycle group, because that grouping
is the thing a reader most needs and the thing a purely mechanical layout
would destroy: app tables are the source of truth, `sync_` is a disposable
mirror, `job_` is queue bookkeeping.
"""

from __future__ import annotations

import asyncio
import html
import pathlib
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "diagrams" / "er.drawio"

#: Left to right: what a request writes, what a sync writes, what a worker
#: writes. Anything unlisted lands in the app column, which is the safe default
#: — a new table is far more likely to be app data than a mirror.
COLUMNS: dict[str, tuple[str, ...]] = {
    # Split across two columns rather than one ten-table tower: edges between
    # `messages`, `runs`, `node_executions`, `pending_inputs` and `actions` are
    # the densest part of the schema, and stacking them all vertically routes
    # every one of those lines straight through the boxes.
    "App — identity and the thread": (
        "users",
        "oauth_tokens",
        "conversations",
        "messages",
        "conversation_entities",
        "audit_log",
    ),
    "App — one execution": (
        "runs",
        "node_executions",
        "pending_inputs",
        "actions",
    ),
    "sync_ — a disposable mirror": (
        "sync_messages",
        "sync_events",
        "sync_files",
        "sync_state",
    ),
    "job_ — queue bookkeeping": ("job_failed_tasks",),
}

#: Short type names. `character varying(255)` in a box is noise; `varchar` is
#: the fact somebody needs.
TYPES = {
    "character varying": "varchar",
    "character": "char",
    "timestamp with time zone": "timestamptz",
    "timestamp without time zone": "timestamp",
    "double precision": "float8",
    "boolean": "bool",
    "integer": "int",
    "bigint": "bigint",
    "smallint": "smallint",
    "jsonb": "jsonb",
    "uuid": "uuid",
    "text": "text",
    "ARRAY": "[]",
    "USER-DEFINED": "enum",
}

HEADER_H = 26
ROW_H = 18
WIDTH = 260
GAP_X = 340
GAP_Y = 40
TOP = 60

STYLE_TABLE = (
    "shape=table;startSize={h};container=1;collapsible=0;childLayout=tableLayout;"
    "fixedRows=1;rowLines=0;fontStyle=1;align=center;resizeLast=1;"
    "fillColor={fill};strokeColor={stroke};fontColor=#1F2933;"
)
STYLE_ROW = (
    "shape=tableRow;horizontal=0;startSize=0;swimlaneHead=0;swimlaneBody=0;"
    "fillColor=none;collapsible=0;dropTarget=0;points=[[0,0.5],[1,0.5]];"
    "portConstraint=eastwest;strokeColor=none;top=0;left=0;right=0;bottom=0;"
)
STYLE_CELL = (
    "shape=partialRectangle;overflow=hidden;connectable=0;fillColor=none;"
    "align=left;verticalAlign=middle;spacingLeft=8;spacingRight=8;"
    "top=0;left=0;bottom=0;right=0;fontSize=11;fontColor=#3E4C59;"
)
STYLE_EDGE = (
    "edgeStyle=entityRelationEdgeStyle;rounded=0;html=1;endArrow=ERoneToMany;"
    "startArrow=ERmandOne;strokeColor={colour};strokeWidth=1.4;exitX=1;exitY=0.5;"
    "entryX=0;entryY=0.5;"
)

PALETTE = {
    "App — identity and the thread": ("#EEF2FF", "#4F46E5"),
    "App — one execution": ("#EEF2FF", "#4F46E5"),
    "sync_ — a disposable mirror": ("#ECFDF5", "#059669"),
    "job_ — queue bookkeeping": ("#FEF3C7", "#B45309"),
}


def short_type(data_type: str, nullable: str) -> str:
    name = TYPES.get(data_type, data_type)
    return name if nullable == "NO" else f"{name}?"


async def load() -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]], set[tuple[str, str]]]:
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.connect() as conn:
        names = (
            await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname='public' AND tablename <> 'alembic_version' "
                    "ORDER BY tablename"
                )
            )
        ).scalars().all()

        tables: dict[str, list[tuple[str, str]]] = {}
        for name in names:
            rows = (
                await conn.execute(
                    text(
                        "SELECT column_name, data_type, is_nullable "
                        "FROM information_schema.columns "
                        "WHERE table_name = :t ORDER BY ordinal_position"
                    ),
                    {"t": name},
                )
            ).all()
            tables[name] = [(c, short_type(t, n)) for c, t, n in rows]

        pks = {
            (t, c)
            for t, c in (
                await conn.execute(
                    text(
                        """
                        SELECT tc.table_name, kcu.column_name
                          FROM information_schema.table_constraints tc
                          JOIN information_schema.key_column_usage kcu
                            ON tc.constraint_name = kcu.constraint_name
                         WHERE tc.constraint_type = 'PRIMARY KEY'
                           AND tc.table_schema = 'public'
                        """
                    )
                )
            ).all()
        }

        fks = [
            (a, col, b)
            for a, col, b in (
                await conn.execute(
                    text(
                        """
                        SELECT tc.table_name, kcu.column_name, ccu.table_name
                          FROM information_schema.table_constraints tc
                          JOIN information_schema.key_column_usage kcu
                            ON tc.constraint_name = kcu.constraint_name
                          JOIN information_schema.constraint_column_usage ccu
                            ON ccu.constraint_name = tc.constraint_name
                         WHERE tc.constraint_type = 'FOREIGN KEY'
                           AND tc.table_schema = 'public'
                        """
                    )
                )
            ).all()
        ]
    await engine.dispose()
    return tables, fks, pks


#: How many non-key columns a box shows before it summarises the rest.
#:
#: A diagram is a map, not the DDL. Printing all twenty-five columns of
#: `sync_messages` makes the box tall enough that every relationship line has
#: to cross it, and the crossings are what make the picture unreadable. Keys
#: and a few identifying fields say what the table *is*; `docs/schema.md` says
#: what it holds.
DETAIL_ROWS: Final[int] = 5

#: Columns worth showing beyond the keys, because they say what a row is.
SALIENT = (
    "email", "subject", "title", "name", "status", "kind", "role", "op",
    "connector", "service", "sent_at", "starts_at", "modified_at",
    "created_at", "occurred_at", "seq", "action",
)


def condense(
    table: str,
    columns: list[tuple[str, str]],
    keys: set[str],
) -> list[tuple[str, str, bool]]:
    """Keys, then a few telling columns, then how many were left out."""
    shown: list[tuple[str, str, bool]] = []
    for name, typ in columns:
        if name in keys:
            shown.append((name, typ, True))

    extras = [
        (n, t) for n, t in columns if n not in keys and n in SALIENT
    ][:DETAIL_ROWS]
    shown.extend((n, t, False) for n, t in extras)

    hidden = len(columns) - len(shown)
    if hidden > 0:
        shown.append((f"+{hidden} more", "", False))
    return shown


def render(
    tables: dict[str, list[tuple[str, str]]],
    fks: list[tuple[str, str, str]],
    pks: set[tuple[str, str]],
) -> str:
    cells: list[str] = []
    placed: set[str] = set()
    uid = [1]

    def nid() -> str:
        uid[0] += 1
        return f"n{uid[0]}"

    table_ids: dict[str, str] = {}

    for col_index, (group, names) in enumerate(COLUMNS.items()):
        fill, stroke = PALETTE[group]
        x = 40 + col_index * GAP_X
        y = TOP

        cells.append(
            f'<mxCell id="{nid()}" value="{html.escape(group)}" '
            f'style="text;html=1;fontSize=14;fontStyle=1;fontColor={stroke};align=left;" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="20" '
            f'width="{WIDTH}" height="24" as="geometry"/></mxCell>'
        )

        for name in names:
            raw = tables.get(name)
            if raw is None:
                continue
            placed.add(name)
            keys = {c for t, c in pks if t == name} | {
                col for src, col, _ in fks if src == name
            }
            columns = condense(name, raw, keys)
            height = HEADER_H + ROW_H * len(columns)
            tid = nid()
            table_ids[name] = tid
            style = STYLE_TABLE.format(h=HEADER_H, fill=fill, stroke=stroke)
            cells.append(
                f'<mxCell id="{tid}" value="{html.escape(name)}" style="{style}" '
                f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
                f'width="{WIDTH}" height="{height}" as="geometry"/></mxCell>'
            )
            for i, (cname, ctype, is_key) in enumerate(columns):
                rid = nid()
                cells.append(
                    f'<mxCell id="{rid}" value="" style="{STYLE_ROW}" vertex="1" '
                    f'parent="{tid}"><mxGeometry y="{HEADER_H + i * ROW_H}" '
                    f'width="{WIDTH}" height="{ROW_H}" as="geometry"/></mxCell>'
                )
                text_label = f"{cname}  ·  {ctype}" if ctype else cname
                label = html.escape(text_label)
                cell_style = STYLE_CELL + ("fontStyle=1;fontColor=#1F2933;" if is_key else "")
                cells.append(
                    f'<mxCell id="{nid()}" value="{label}" style="{STYLE_CELL}" '
                    f'vertex="1" parent="{rid}"><mxGeometry width="{WIDTH}" '
                    f'height="{ROW_H}" as="geometry"><mxRectangle width="{WIDTH}" '
                    f'height="{ROW_H}" as="alternateBounds"/></mxGeometry></mxCell>'
                )
            y += height + GAP_Y

    missing = sorted(set(tables) - placed)
    if missing:  # a table nobody assigned to a column still has to appear
        x = 40 + len(COLUMNS) * GAP_X
        y = TOP
        for name in missing:
            keys = {c for t, c in pks if t == name} | {
                col for src, col, _ in fks if src == name
            }
            columns = condense(name, tables[name], keys)
            height = HEADER_H + ROW_H * len(columns)
            tid = nid()
            table_ids[name] = tid
            style = STYLE_TABLE.format(h=HEADER_H, fill="#F1F5F9", stroke="#64748B")
            cells.append(
                f'<mxCell id="{tid}" value="{html.escape(name)}" style="{style}" '
                f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
                f'width="{WIDTH}" height="{height}" as="geometry"/></mxCell>'
            )
            y += height + GAP_Y

    for source, column, target in sorted(fks):
        a, b = table_ids.get(source), table_ids.get(target)
        if not a or not b:
            continue
        colour = "#94A3B8" if source.startswith(("sync_", "job_")) else "#6366F1"
        cells.append(
            f'<mxCell id="{nid()}" value="{html.escape(column)}" '
            f'style="{STYLE_EDGE.format(colour=colour)}" edge="1" parent="1" '
            f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>'
        )

    body = "\n        ".join(cells)
    return (
        '<mxfile host="app.diagrams.net">\n'
        '  <diagram name="Schema">\n'
        '    <mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="1600" pageHeight="2400" math="0" shadow="0">\n'
        "      <root>\n"
        '        <mxCell id="0"/>\n'
        '        <mxCell id="1" parent="0"/>\n'
        f"        {body}\n"
        "      </root>\n"
        "    </mxGraphModel>\n"
        "  </diagram>\n"
        "</mxfile>\n"
    )


async def main() -> None:
    tables, fks, pks = await load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(tables, fks, pks))
    print(f"  wrote {OUT}")
    print(f"  {len(tables)} tables, {len(fks)} foreign keys")
    print(f"  drawio -x -f png --scale 2.5 -o er.png {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
