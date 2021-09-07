import tweepy, json


consumer_key = ""
consumer_secret = ""

access_token = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# cuenta
#data = api.get_user("nike")
#print (data)

numero = 0

for tweet in tweepy.Cursor(api.search , q = "sequía", tweet_mode = "extended").items(10):

    hola = str(json.dumps(tweet._json["full_text"], ensure_ascii=False, indent=2))

    if hola[0:3] == '"RT':
        hola2 = str(json.dumps(tweet._json["retweeted_status"]["full_text"], ensure_ascii=False, indent=2))
        print ("RT = " + hola2)

    else:
        print ("Tweet = " + hola)