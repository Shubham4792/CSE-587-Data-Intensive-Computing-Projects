
# Project done by SHUBHAM SHAILESH PANDEY, UBID - spandey4


```python
from nytimesarticle import articleAPI
api = articleAPI('1f229c0375cf4155ab7bf894ecea4ff8')
```


```python
tmp = []
```

### The following cell searches for NYTimes articles containing the word technology(as an example here) and prints out the URLS returned by the API. These URLs are rejected if they are duplicate or if they do not belong to technology category(as an example here)


```python
articles = api.search(q='technology',page = 1)
for i in range(10):
    x = articles['response']['docs'][i]['web_url']
    print(x)
    if('technology' in x):
        tmp.append(x)
    tmp = list(set(tmp))
print(len(tmp))
```

    https://www.nytimes.com/2018/03/23/business/elder-orphans-care.html
    https://www.nytimes.com/2018/03/22/business/shkreli-holmes-fraud.html
    https://www.nytimes.com/2018/03/19/technology/facebook-alex-stamos.html
    https://www.nytimes.com/2018/03/15/business/with-one-battle-over-a-bigger-one-looms-for-qualcomm-apple.html
    https://www.nytimes.com/2018/03/08/business/tariff-trump-trade-wars.html
    https://www.nytimes.com/2018/03/04/technology/fake-videos-deepfakes.html
    https://www.nytimes.com/2018/02/19/technology/ai-researchers-desks-boss.html
    https://www.nytimes.com/2018/02/15/business/inflation-stocks-interest-rates.html
    https://www.nytimes.com/2018/02/09/technology/farhad-week-tech-homepod-elon-musk.html
    https://www.nytimes.com/2018/01/28/technology/side-benefit-to-amazons-headquarters-contest-local-expertise.html
    85
    

### The URLS collected from the step above are used by BeautifulSoup library and the data is scraped from it and written to a file


```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
count = 0
for k in range(len(tmp)):
    count = count+1
    html = urlopen(tmp[k])
    soup = BeautifulSoup(html)
    all_data = soup.find_all("p", class_='e2kc3sl0')
    if len(all_data) == 0 :
        all_data = soup.find_all("p", class_='story-body-text story-content')
    x = ''
    for i in range(len(all_data)):
        text = ''
        for j in all_data[i].contents:
            if j.string is not None:
                text += j.string
        x += text
    filename = "./Unknown/Technology/Technology"+ str(count)+ ".txt"    
    with open(filename, "w", encoding="utf-8") as myfile:
        myfile.write(x)
```

    D:\Programs\Anaconda3\lib\site-packages\bs4\__init__.py:181: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    The code that caused this warning is on line 193 of the file D:\Programs\Anaconda3\lib\runpy.py. To get rid of this warning, change code that looks like this:
    
     BeautifulSoup(YOUR_MARKUP})
    
    to this:
    
     BeautifulSoup(YOUR_MARKUP, "lxml")
    
      markup_type=markup_type))
    
