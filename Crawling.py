# 新闻文本爬取
import requests
from bs4 import BeautifulSoup
import pandas as pd
 
 
def get_html_text(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except Exception as e:
        print(e)
        print(url)
        return url
 
 
def parse_news_page(html):
    try:
        ilt = []
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title").string
        ilt.append(title)
        content = soup.find_all("p")
        for p in content:
            s = p.text.strip()
            s = "".join(s.split("\n"))
            ilt.append(s)
        news = "".join(ilt)
        return news
    except Exception as e:
        return e
 
 
def parse_href_page(html, hrefs):
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all("a")
    for tag in tags:
        href = tag.attrs["href"]
        if "shtml" == href[-5:] and href not in hrefs:
            hrefs.append(href)
    return hrefs
 
 
def get_newses(url, newses, labels, count):
    hrefs = []
    html = get_html_text(url)
    parse_href_page(html, hrefs)
    for href in hrefs:
        html = get_html_text(href)
        if html == href:
            continue
        news = parse_news_page(html)
        # print(news)
        newses.append(news)
        labels.append(count)
 
 
def main():
    newses = []
    labels = []
    urls = ["http://finance.cnr.cn/", "http://tech.cnr.cn/", "http://food.cnr.cn/",
            "http://health.cnr.cn/", "http://edu.cnr.cn/", "http://travel.cnr.cn/",
            "http://military.cnr.cn/", "http://auto.cnr.cn/", "http://house.cnr.cn/",
            "http://gongyi.cnr.cn/"]
    count = 0
    for url in urls:
        print(url)
        get_newses(url, newses, labels, count)
        count += 1
    newses = pd.DataFrame({"label": labels, "text": newses})
    newses.to_csv("newses.csv", index=False)
 
 
main()