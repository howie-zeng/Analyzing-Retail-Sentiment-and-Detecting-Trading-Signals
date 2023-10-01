import requests
from bs4 import BeautifulSoup

url = "https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new"

response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"})

# if the request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    comment_elements = soup.find_all("div", class_="Comment")

    for comment in comment_elements:
        comment_text = comment.find("div", class_="Comment-body").text.strip()
        print(comment_text)
        print("Done")
else:
    print("Failed to retrieve the webpage. Status code:", response.status_code)
