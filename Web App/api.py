import flask
from flask import jsonify ,request,render_template
import google.cloud.translate as translate
import google.cloud.datastore as datastore
import os
from wtforms import Form, TextAreaField, validators
import pickle
import nltk
import requests
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request 



app = flask.Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)

try:
    XcountVectorizer = open(os.path.join(cur_dir, 'vectorizer.pkl'), 'rb')
    cv = pickle.load(XcountVectorizer)
finally:
    XcountVectorizer.close()

try:
    d = open(os.path.join(cur_dir, 'sgd_search.pkl'), 'rb')
    sgd_search = pickle.load(d)
finally:
    d.close()

def classify(document):
	textcountv = cv.transform([document])
	pred = sgd_search.predict(textcountv)
	print(type(pred))
	label1=pred[0].upper()
	if(pred == "['entertainment']"):
		label1='ENTERTAINMAINT'
	return label1

def fetchData():
	#fetching the content from the URL
	#
	urllink=[]
	urllink=fetchData1()
	#parsing the URL content and storing in a variable
	article_content_list=[]
	header_list=[]
	
	for link in urllink:
		fetched_data = urllib.request.urlopen(link)
		article_read = fetched_data.read()	
		article_parsed = BeautifulSoup.BeautifulSoup(article_read,'html.parser')
		#returning <p> tags
		title = article_parsed.find("div", id = "orb-modules")
		t = title.find("div", class_ ="story-body")
		for w in title.find_all("h1", class_ = "story-body__h1"):
			header=w.text
		paragraphs = article_parsed.find_all('p')
		article_content = ''
		#looping through the paragraphs and adding them to the variable
		for p in paragraphs:  
			article_content += p.text
		article_content_list.append(article_content)
		header_list.append(header)
	return article_content_list,header_list,urllink

def fetchData1():
	url = "http://www.bbc.com/news"
	# Getting the webpage, creating a Response object.
	response = requests.get(url)
	# Extracting the source code of the page.
	data = response.text
	# Passing the source code to BeautifulSoup to create a BeautifulSoup object for it.
	soup = BeautifulSoup.BeautifulSoup(data, 'lxml')
	#container = soup.find("div", id = "u8996571333555776")
	contain = soup.find("div", class_ = "gel-wrap gs-u-pt+")
	# Extracting all the <a> tags into a list.
	tags = contain.find_all('a')
	print(type(tags))
	urllist=[] 
	# Extracting URLs from the attribute href in the <a> tags.
	for tag in tags:
		abc=tag.get('href')
		if(abc not in urllist):
			index = abc.rfind('-')
			if abc[index+1:].isdigit():
				urllist.append(abc)
	print(urllist)
	urllink = []
	for b in range(0,len(urllist)):
		urllink.append('http://www.bbc.com' + urllist[b])
	print(urllink)
	return urllink[:6]

def _create_dictionary_table(text_string) -> dict:
   
    #removing stop words
    stop_words = set(stopwords.words("english"))
    
    words = word_tokenize(text_string)
    
    #reducing words to their root form
    stem = PorterStemmer()
    
    #creating dictionary for the word frequency table
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table

def _calculate_sentence_scores(sentences, frequency_table) -> dict:   

    #algorithm for scoring a sentence by its words
    sentence_weight = dict()

    for sentence in sentences:
        sentence_wordcount = (len(word_tokenize(sentence)))
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:7] in sentence_weight:
                    sentence_weight[sentence[:7]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:7]] = frequency_table[word_weight]

        sentence_weight[sentence[:7]] = sentence_weight[sentence[:7]] / sentence_wordcount_without_stop_words

    return sentence_weight


def _calculate_average_score(sentence_weight) -> int:
   
    #calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]
	#getting sentence average value from source text
    average_score = (sum_values / len(sentence_weight))
    return average_score

def _get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''

    for sentence in sentences:
        if sentence[:7] in sentence_weight and sentence_weight[sentence[:7]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1
    return article_summary


def _run_article_summary(article):
    
    #creating a dictionary for the word frequency table
    frequency_table = _create_dictionary_table(article)

    #tokenizing the sentences
    sentences = sent_tokenize(article)

    #algorithm for scoring a sentence by its words
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)

    #getting the threshold
    threshold = _calculate_average_score(sentence_scores)

    #producing the summary
    article_summary = _get_article_summary(sentences, sentence_scores, 1.25 * threshold)

    return article_summary


######## Flask
class ReviewForm(Form):
	lyrics = TextAreaField('',[validators.DataRequired(), validators.length(min=15)])

@app.route('/api', methods=['GET', 'POST'])
def home():
	form = ReviewForm(request.form)
	return render_template("dummy.html",form=form)

@app.route('/api/newsclasifiy',methods=['POST'])
def api_classify():
	form = ReviewForm(request.form)	
	if request.method == 'POST' and request.form['operation']=='CLASSIFY':
		if form.validate():
			text=request.form['lyrics']
			lyricsgiven=request.form['lyrics']
			label= classify(text)
		else:
			return "Text field empty"
	elif request.method =='POST' and request.form['operation']=='BACK':
		form = ReviewForm()
		return render_template("dummy.html",form=form)	
	return render_template("dummy.html",form=form,label=label)
	
@app.route('/api/newsextract',methods=['POST'])
def api_summerization():
	try:
		form = ReviewForm(request.form)
		article_content_list=[]
		article_content_list,header_list,urllink=fetchData()
		summary_results_list=[]
		new_header_list=[]
		new_urllink=[]
		print(len(article_content_list))
		for i in range(0,len(article_content_list)):
			if isinstance(article_content_list[i],str) and isinstance(header_list[i],str) and article_content_list[i] :
				summary_results = _run_article_summary(article_content_list[i])
				summary_results_list.append(summary_results)
				new_header_list.append(header_list[i])
				new_urllink.append(urllink[i])
		print(len(summary_results_list))
		print(len(new_header_list))
		header_summary_link=zip(new_header_list,summary_results_list,new_urllink)
		return render_template("summary.html",header_summary=header_summary_link)
	except:
		msg="Something went wrong !!! Try agaian"
		return render_template("summary.html",msg=msg)
		
	
if __name__ == '__main__':
	app.run(debug=True)