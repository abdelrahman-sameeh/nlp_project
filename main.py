import pandas as pd
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


nlp = spacy.load(
    "en_core_web_sm"
)  # English tokenizer, tagger, parser, NER, and word vectors
stop_words = set(stopwords.words("english"))  # Set of English stopwords
ps = PorterStemmer()  # Initialize the Porter stemmer

print("stop words=", stop_words)
print("\n", "###" * 20, '\n')


# Function to process text
def process_text(text):
    # 1. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # 2. Filter stop words
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # 3. Stemming
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

    # 4. Lemmatizing
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    # 5. POS Tagging
    pos_tags = nltk.pos_tag(lemmatized_tokens)

    # 6. Chunking
    chunked = nltk.ne_chunk(pos_tags)

    # 7. Chinking (remove certain chunks) JJ>>صفه, NN>>اسم
    grammar = "NP: {<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)

    # 8. Named-Entity Recognition (date, countries) تاريخ او اسم علم زى مصر
    ner = [(ent.text, ent.label_) for ent in doc.ents]

    # 9. Dependency Parsing
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

    # 10. Rule-Based Matching
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "school"}]
    matches = f'No results match this pattern {pattern}'
    try:
        matcher.add("SchoolPattern", [pattern])
        matches = matcher(doc)
    except:
        pass

    # 11. Vectorization
    vectorizer = CountVectorizer()
    vectorized_text = vectorizer.fit_transform([" ".join(lemmatized_tokens)])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized_text = tfidf_vectorizer.fit_transform(
        [" ".join(lemmatized_tokens)]
    )

    return {
        "removed_punctuations": text,
        "filtered_tokens": filtered_tokens,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "pos_tags": pos_tags,
        "chunked": chunked,
        "tree(Chinking)": tree,
        "ner": ner,
        "dependencies": dependencies,
        "matches": matches,
        "vectorized_text": vectorized_text.toarray(),
        "tfidf_vectorized_text": tfidf_vectorized_text.toarray(),
    }


# Load the CSV file
df = pd.read_csv("school_reviews.csv")


for index in range(1):
    for key, value in process_text(df['Review'][index]).items():
        print(f"{key}: {value}\n")
    print("#" * 80)
