from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

import time
from bs4 import BeautifulSoup

driver = webdriver.Chrome(ChromeDriverManager().install())

search_url = "https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new"
driver.get(search_url)

# scroll down the page 5 times
for _ in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # wait

html = driver.page_source

soup = BeautifulSoup(html, "html.parser")

comment_elements = soup.find_all("div", class_="Comment")

for comment in comment_elements:
    comment_text = comment.find("div", class_="Comment__body").text.strip()
    print(comment_text)

driver.quit()

