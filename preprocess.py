import spacy
import read_pdf

job_desc = ""
with open("Job Description\job_desc.txt","r") as f:
    line = f.read()
    job_desc+=line

# load english language model and create nlp object from it
nlp = spacy.load("en_core_web_lg") 
'''
D:\Sem-7\Btech_Project\FinalYear_Project\Code\Backend\Resume Ranking System\similarity.py:9: UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
  sim_score = resume_doc.similarity(job_doc)
'''

def preprocess(text):
    # remove stop words and lemmatize the text
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            # print(token)
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens) 

preprocessed_job_desc = preprocess(job_desc)

preprocessed_pdf_text = []

for text in read_pdf.text_list:

    temp = preprocess(text)

    preprocessed_pdf_text.append(temp)



