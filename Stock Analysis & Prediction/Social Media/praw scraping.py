import csv
import praw
import datetime
import pandas as pd

# Reddit API credentials
# https://www.reddit.com/prefs/apps
client_id = 'ZJsB8bfPDyqFgtIEQll70Q'
client_secret = 'oKVpPC8ycIeGRk8uYWp38WgM31WENg'
user_agent = 'myapp'

reddit = praw.Reddit(client_id = client_id, 
                     client_secret = client_secret, 
                     user_agent = user_agent) 

subreddit_name = 'wallstreetbets'
search_query = 'flair:"Daily Discussion"'
sort_method = 'new'  # 'hot', 'top', 'rising'

subreddit = reddit.subreddit(subreddit_name)

post_list = [[s.id, s.url, s.title, s.num_comments, s.score, s.author, s.created, s.selftext] for s in subreddit.top(limit = 10)]
post_df = pd.DataFrame(post_list, columns = ["id", "url", "title", "num_comments", "score", "author", "created", "selftext"])
post_df.to_csv('post_data.csv', index=False)

# sub = reddit.submission(id = 'l7feld')
# sub.comments.replace_more(limit = 10)
# comment_list = [[c.id, c.author, c.body, c.score, c.created] for c in sub.comments]
# comment_df = pd.DataFrame(comment_list, columns = ["id", "author", "created", "score", "selftext"])
# comment_df.to_csv('comment_data.csv', index=False)
# print(comment_df)

posts = subreddit.search(search_query, sort=sort_method)

# output_file = 'reddit_comments.txt'
# with open(output_file, 'w', encoding='utf-8') as file:
#     for post in posts:
#         file.write(f"Post Title: {post.title}\n")
#         file.write(f"Post URL: {post.url}\n")
#         file.write("Comments:\n")

#         # accesses and saves up to 10 comments for each post
#         post.comments.replace_more(limit=10)
#         for comment in post.comments.list():
#             file.write(comment.body + '\n')

#         file.write("=" * 50 + '\n')
# print(f"Comments saved to {output_file}")

output_file = 'reddit_comments.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Title', 'URL', 'Date', 'Comments'])
    
    for post in posts:
        title = post.title
        url = post.url
        post_date = post.created_utc
        post_date = datetime.datetime.utcfromtimestamp(post_date).strftime('%Y-%m-%d %H:%M:%S')
        
        post.comments.replace_more(limit=10)
        # comments = '\n'.join([comment.body for comment in post.comments.list()])
        comments = [comment.body for comment in post.comments.list()]
        
        csv_writer.writerow([title, url, post_date, comments])

print(f"Comments saved to {output_file}")