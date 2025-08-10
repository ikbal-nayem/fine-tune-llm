from docling.document_converter import DocumentConverter

import urllib.request
from io import BytesIO
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

source = "https://cdn3.ogrlegal.com/files/land/act/Bhumi-Khatian_Chittagong_Hill_Tracts_Ordinance_1984.pdf"
url = "http://bdlaws.minlaw.gov.bd/act-print-1122.html"


def main():
    # converter = DocumentConverter()
    # doc = converter.convert(source).document

    # print(doc.export_to_markdown())
    # with open('output-data/ooo.md', 'w') as f:
    #     f.write(doc.export_to_markdown())

    html_content = urllib.request.urlopen(url).read()
    in_doc = InputDocument(
        path_or_stream=BytesIO(html_content),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="downloaded_page.html", # Optional: provide a filename
    )
    backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(html_content))
    dl_doc = backend.convert()
    markdown_output = dl_doc.export_to_markdown() # Or .export_to_json()
    print(markdown_output)

    with open('output-data/ooo2.txt', 'w') as f:
        f.write(markdown_output)


if __name__ == "__main__":
    main()
