import cohere
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS 
nltk.download("punkt")
co = cohere.Client('YOuR KEY')#add your key here
model = api.load("glove-wiki-gigaword-50")
def get_similar(prompt, top_n=3):
 words = word_tokenize(prompt.lower())
 words = [w for w in words if w not in STOPWORDS and w.isalpha()]
 sims = []
 for w in words:
 if w in model:
 sims += [s[0] for s in model.most_similar(w, topn=top_n)]
 return " ".join(set(sims))
def generate(prompt):
 res = co.generate(model="command", prompt=prompt)
 return res.generations[0].text.strip()
prompt = "Describe the recent match won by RCB in IPL."#add any prompt
enriched = prompt + " " + get_similar(prompt)
print("Original Prompt:\n", prompt)
print("\nResponse:\n", generate(prompt))
print("\nEnriched Prompt:\n", enriched)
print("\nEnriched Response:\n", generate(enriched))
