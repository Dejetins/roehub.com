"""Serialize a complete trades CSV as an XLSX workbook without executing cell formulas."""

import csv
import io
from decimal import Decimal, InvalidOperation
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_TEXT_FIELDS = {"entry_timestamp", "exit_timestamp", "side", "exit_reason", "timeframe"}


def trades_xlsx(content: str) -> bytes:
    """Keep every row, splitting Excel's sheet limit across sheets in one file."""
    source = csv.reader(io.StringIO(content))
    headers = next(source, [])
    output = io.BytesIO()
    names = []
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        sheet = None
        row_number = 0
        for row in source:
            if sheet is None or row_number >= 1_048_576:
                if sheet is not None:
                    sheet.write(b"</sheetData></worksheet>")
                    sheet.close()
                names.append(f"Trades {len(names) + 1}")
                sheet = archive.open(f"xl/worksheets/sheet{len(names)}.xml", "w")
                sheet.write(
                    b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                )
                sheet.write(_row(headers, headers, header=True).encode())
                row_number = 1
            sheet.write(_row(row, headers).encode())
            row_number += 1
        if sheet is not None:
            sheet.write(b"</sheetData></worksheet>")
            sheet.close()
        else:
            names.append("Trades 1")
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml'
                '/2006/main"><sheetData>'
                + _row(headers, headers, header=True)
                + "</sheetData></worksheet>",
            )
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/cont'
            'ent-types"><Default Extension="rels" ContentType="application/vnd'
            '.openxmlformats-package.relationships+xml"/><Default Extension="x'
            'ml" ContentType="application/xml"/><Override PartName="/xl/workbo'
            'ok.xml" ContentType="application/vnd.openxmlformats-officedocumen'
            't.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="app'
                f'lication/vnd.openxmlformats-officedocument.spreadsheetml.workshee'
                f't+xml"/>'
                for i in range(1, len(names) + 1)
            )
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2'
            '006/relationships"><Relationship Id="rId1" Type="http://schemas.o'
            'penxmlformats.org/officeDocument/2006/relationships/officeDocumen'
            't" Target="xl/workbook.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
            '2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocum'
            'ent/2006/relationships"><sheets>'
            + "".join(
                f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
                for i, name in enumerate(names, 1)
            )
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org'
                f'/officeDocument/2006/relationships/worksheet" Target="worksheets/'
                f'sheet{i}.xml"/>'
                for i in range(1, len(names) + 1)
            )
            + "</Relationships>",
        )
    return output.getvalue()


def _row(values: list[str], headers: list[str], *, header: bool = False) -> str:
    cells = []
    for index, value in enumerate(values):
        numeric = False
        if not header and index < len(headers) and headers[index] not in _TEXT_FIELDS:
            try:
                numeric = Decimal(value).is_finite()
            except InvalidOperation:
                pass
        if numeric:
            cells.append(f'<c t="n"><v>{escape(value)}</v></c>')
        else:
            clean = "".join(c for c in value if ord(c) >= 32 or c in "\t\n\r")
            cells.append(
                f'<c t="inlineStr"><is><t xml:space="preserve">{escape(clean)}</t></is></c>'
            )
    return "<row>" + "".join(cells) + "</row>"
