import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp(u"Hoy aún tiene tiempo para quedar en la historia de Chile como un Estadista que guió a su pueblo por la senda de la Democracia y no lo entregó a una dictadura marxista.  Vivimos desesperanzados, hemos perdido la fe.  Por favor reacciones!! Devuélvanos la esperanza !!")
for sent in doc.sents:
    displacy.render(nlp(sent.text).style='ent'
    #,jupyter=True
    )