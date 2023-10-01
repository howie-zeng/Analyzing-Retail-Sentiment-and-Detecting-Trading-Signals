import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup

url = 'https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new'

# Initialize the Selenium WebDriver
# chromedriver_path = r"D:\Cornell\course\CS6386\Analyzing-the-Correlation-Between-Retail-Traders--Sentiments-and-Equity-Market-Movements\chromedriver.exe"
# chrome_service = ChromeService(chromedriver_path)
# driver = webdriver.Chrome(service=chrome_service)

chrome_options = Options()
# chrome_options.add_argument('--headless')
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)

def scroll_and_load_comments(driver):
    for _ in range(5): 
        actions = ActionChains(driver)
        actions.send_keys(Keys.END)
        actions.perform()
        time.sleep(2) 

scroll_and_load_comments(driver)

# HTML source of the page
page_source = driver.page_source

soup = BeautifulSoup(page_source, 'html.parser')

comments = soup.find_all('div', {'class': 'Comment'})
for comment in comments:
    comment_text = comment.find('div', {'class': 'Comment__body'}).text
    print(comment_text)
    
driver.quit()
