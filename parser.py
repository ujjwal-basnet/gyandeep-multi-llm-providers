import pymupdf

pdf = pymupdf.open("./pdfs/grade-5-science-and-technology.pdf")

total_book = ""
page_count = 1
total_pages = pdf.page_count

for page in pdf:
    print(f"On page: {page_count} | {round((page_count/total_pages) * 100)}%")
    total_book += page.get_textpage_ocr().extractText()
    page_count += 1

with open("totalBook.txt", "w") as book_text:
    book_text.write(total_book)

