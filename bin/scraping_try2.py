from collections import Counter
import numpy as np
from datetime import date, timedelta
from dateutil.parser import parse 

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def grab_html():
    url = 'https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new'
    chrome_driver_path = 'chromedriver.exe'
    driver = webdriver.Chrome(executable_path=chrome_driver_path)
    
    chrome_options = Options()
    # chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    return driver

def grab_link(driver):
    yesterday = date.today() - timedelta(days=1)
    links = driver.find_elements_by_xpath('//[@class="_eYtD2XCVieq6emjKBH3m"]') 
    for a in links:
        if a.text.startswith('Daily Discussion Thread'):
             date_str  = ''.join(a.text.split(' ')[-3:])
             parsed = parse(date_str) 
             if parse(str(yesterday)) == parsed:
                link = a.find_element_by_xpath('../..').get_attribute('href')
        if a.text.startswith('Weekend'):
             weekend_date = a.text.split(' ')
             parsed_date = weekend_date[-3] + ' ' + weekend_date[-2].split('-')[1] + weekend_date[-1] 
        parsed = parse(parsed_date) 
        saturday = weekend_date[-3] + ' ' + str(int(weekend_date[-2].split('-')[1].replace(',','')) - 1) + ' ' + weekend_date[-1]
        
        if parse(str(yesterday)) == parsed: 
            link = a.find_element_by_xpath('../..').get_attribute('href')
        elif parse(str(yesterday)) == parse(str(saturday)):
            link = a.find_element_by_xpath('../..').get_attribute('href') 
        
        stock_link = link.split('/')[-3]
    
    driver.close()
    
    return stock_link

def grab_commentid_list(stock_link):
    html = requests.get(f'https://api.pushshift.io/reddit/submission/comment_ids/{stock_link}')
    raw_comment_list = html.json()
    return raw_comment_list
    

def grab_stocklist():
    with open('stockslist.txt', 'r') as w:
        stocks = w.readlines()
        stocks_list = []
        for a in stocks:
            a = a.replace('\n','')
            stocks_list.append(a)
    return stocks_list

def get_comments(comment_list):
    html = requests.get(f'https://api.pushshift.io/reddit/comment/search?ids={comment_list}&fields=body&size=1000')
     
    newcomments = html.json()
    return newcomments

def get_stock_list(newcomments,stocks_list):
    stock_dict = Counter()
    for a in newcomments['data']:
        for ticker in stocks_list:
            if ticker in a['body']:
                stock_dict[ticker]+=1
    return stock_dict

def grab_stock_count(stock_dict,raw_comment_list):
     orig_list = np.array(raw_comment_list['data'])
     comment_list = ",".join(orig_list[0:1000])
     remove_me = slice(0,1000)
     cleaned = np.delete(orig_list, remove_me)
     i = 0
     while i < len(cleaned):
        print(len(cleaned))
        cleaned = np.delete(cleaned, remove_me)
        new_comments_list = ",".join(cleaned[0:1000])
        newcomments = get_comments(new_comments_list)
        get_stock_list(newcomments,stocks_list)
     stock = dict(stock_dict) 
     return stock



if __name__ == "__main__":
    driver = grab_html()
    stock_link = grab_link(driver)
    grab_commentid_list(stock_link) 
    stockslist = grab_stocklist()
    newcomments = get_comments(comment_list)
    stock_dict = get_stock_list(new_comments,stocks_list)
    stock = grab_stock_count(stock_dict)