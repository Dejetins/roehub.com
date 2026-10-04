import io
from xml.etree import ElementTree
from zipfile import ZipFile

from apps.api.routes.backtest_export import trades_xlsx


def test_xlsx_keeps_all_rows_and_numeric_values_without_formulas():
    content = "trade_index,exit_reason,net_pnl_quote\n" + "".join(
        f'{i},"=SUM(1;2)",{i}.25\n' for i in range(100_005)
    )
    with ZipFile(io.BytesIO(trades_xlsx(content))) as archive:
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = sheet.findall("x:sheetData/x:row", ns)
        assert len(rows) == 100_006
        index_cell = rows[-1][0].find("x:v", ns)
        pnl_cell = rows[-1][2].find("x:v", ns)
        assert index_cell is not None and pnl_cell is not None
        assert index_cell.text == "100004"
        assert pnl_cell.text == "100004.25"
        assert rows[1][1].attrib["t"] == "inlineStr"
        assert not sheet.findall(".//x:f", ns)
        assert "[Content_Types].xml" in archive.namelist()


def test_empty_export_has_one_sheet_and_headers():
    with ZipFile(io.BytesIO(trades_xlsx("trade_index,side\n"))) as archive:
        assert b"trade_index" in archive.read("xl/worksheets/sheet1.xml")
