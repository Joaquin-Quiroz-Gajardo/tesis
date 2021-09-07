from nltk.corpus import stopwords
from collections import Counter
text = "le dije hola a los perros que corren por la pradera"
stop_words = stopwords.words('spanish')
stop_words.extend(["hola","chao"])
print(stop_words)
stopwords_dict = Counter(stop_words)
text = ' '.join([word for word in text.split() if word not in stopwords_dict])
print(text)