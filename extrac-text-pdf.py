import PyPDF2

f = open('enade.pdf', 'rb')
pdf = PyPDF2.PdfFileReader(f)
for i in range(pdf.getNumPages()):
    print("--- Pagina %d -----------------------------"%i)
    pg = pdf.getPage(i).extractText()
    print(pg)
    print("-------------------------------------------")