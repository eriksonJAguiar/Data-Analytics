import re
import urllib.request
from bs4 import BeautifulSoup
     
html = urllib.request.urlopen('https://www.uol.com.br/')
soup = BeautifulSoup(html)
data = soup.findAll(text=True)

     
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title', '\n', '\t']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True
     
result = filter(visible, data)

#result  = [r for i in result i in ['\t','\n']]

for r in result:
    if not r in ['\n','\t', '', ' ']:
        print(r)