from nltk.corpus import stopwords
from collections import Counter
text = "@jkhxgucci Las jikukas estamos en sequía 😭💖 https://t.co/EjdVanYO9l"
# stop_words = stopwords.words('spanish')
# stop_words.extend(["hola","chao"])
# print(stop_words)
# stopwords_dict = Counter(stop_words)
# text = ' '.join([word for word in text.split() if word not in stopwords_dict])
# print(text)

text = text.encode("ascii", 'ignore')
print(text)