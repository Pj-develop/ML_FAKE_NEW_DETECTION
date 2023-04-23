import requests
from bs4 import BeautifulSoup

def check_false_news(url):
    try:
        # Validate user input
        if not url.startswith("http"):
            url = "http://" + url

        # Fetch the HTML content of the given URL
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Check for indicators of false news
        false_news_keywords = ['fake', 'hoax', 'conspiracy', 'misleading', 'clickbait']
        for keyword in false_news_keywords:
            if keyword in soup.text.lower():
                return True

        return False

    except (requests.exceptions.InvalidURL, requests.exceptions.ConnectionError):
        print("Invalid URL or unable to connect to the website.")
        return None
ur=input("Enter URL: ")
if check_false_news(ur)==False:
    print("news is true")
else:
    print("News is false")
