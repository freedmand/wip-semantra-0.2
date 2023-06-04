import json
from contextlib import contextmanager

import pypdfium2 as pdfium
from ilock import ILock

from util.context import Context

# Page separator character
LINE_FEED = "\f"


@contextmanager
def pdf_document(filename: str):
    pdf = pdfium.PdfDocument(filename)
    try:
        yield pdf
    finally:
        pdf.close()


@contextmanager
def pdf_page(pdf: pdfium.PdfDocument, page_index: int):
    page = pdf[page_index]
    try:
        yield page
    finally:
        page.close()


@contextmanager
def pdf_text_page(page: pdfium.PdfPage):
    textpage = page.get_textpage()
    try:
        yield textpage
    finally:
        textpage.close()


def transform_pdf(
    ctx: Context,
    pdf_filename: str,
    output_txt_filename: str,
    output_page_position_json_filename: str,
):
    with ILock(pdf_filename):
        with pdf_document(pdf_filename) as pdf:
            n_pages = len(pdf)

            positions = []
            position = 0
            with open(
                output_txt_filename, "w", newline="", encoding="utf-8", errors="ignore"
            ) as f:
                for page_index in ctx.progress(
                    range(n_pages), desc="Extracting PDF contents"
                ):
                    with pdf_page(pdf, page_index) as page:
                        page_width, page_height = page.get_size()
                        with pdf_text_page(page) as textpage:
                            pagetext = textpage.get_text_range()

                        positions.append(
                            {
                                "char_index": position,
                                "page_width": page_width,
                                "page_height": page_height,
                            }
                        )
                        position += f.write(pagetext)
                        position += f.write(LINE_FEED)
            with open(output_page_position_json_filename, "w", encoding="utf-8") as f:
                json.dump(positions, f)

            return n_pages
