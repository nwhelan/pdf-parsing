from pdfplay.models import BBox, Block, PageResult, ParseResult, Table, TableCell


def test_bottom_left_conversion_flips_y():
    # A box 100pt tall sitting 700pt above the bottom of a 792pt page starts
    # 92pt from the top.
    box = BBox.from_bottom_left(10, 700, 50, 720, page_height=792)
    assert box.x0 == 10
    assert box.y0 == 72
    assert box.y1 == 92
    assert box.height == 20


def test_normalized_swaps_inverted_corners():
    box = BBox(x0=50, y0=90, x1=10, y1=20).normalized()
    assert (box.x0, box.y0, box.x1, box.y1) == (10, 20, 50, 90)


def test_iou_of_identical_and_disjoint_boxes():
    a = BBox(x0=0, y0=0, x1=10, y1=10)
    assert a.iou(a) == 1.0
    assert a.iou(BBox(x0=20, y0=20, x1=30, y1=30)) == 0.0
    half = BBox(x0=5, y0=0, x1=15, y1=10)
    assert abs(a.iou(half) - (50 / 150)) < 1e-9


def test_table_markdown_round_trip():
    table = Table(
        id="t",
        page=1,
        n_rows=2,
        n_cols=2,
        cells=[
            TableCell(row=0, col=0, text="Date", is_header=True),
            TableCell(row=0, col=1, text="Amount", is_header=True),
            TableCell(row=1, col=0, text="03/01"),
            TableCell(row=1, col=1, text="10.00"),
        ],
    )
    md = table.to_markdown()
    assert md.splitlines()[0] == "| Date | Amount |"
    assert md.splitlines()[2] == "| 03/01 | 10.00 |"


def test_parse_result_layer_ordering_is_coarse_to_fine():
    page = PageResult(
        page_number=1,
        width=612,
        height=792,
        blocks=[
            Block(id="a", page=1, layer="table"),
            Block(id="b", page=1, layer="word"),
            Block(id="c", page=1, layer="line"),
        ],
    )
    result = ParseResult(parser_id="x", pages=[page])
    assert result.layers() == ["word", "line", "table"]
