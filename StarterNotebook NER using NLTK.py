#!/usr/bin/env python
# coding: utf-8

# ## `Berchmans`

# # `Starter Notebook NER using NLTK`

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk


# In[2]:


sentence1 = "Kevin said on Monday that WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Renitta E. Lynch, the top federal prosecutor in Karthikeyan, spoke forcefully about the pain of a broken trust that African-Americans felt and Berchmans said the responsibility for repairing generations of Venkat miscommunication and mistrust fell to law enforcement."


# In[3]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# `The sentence needs to be tokenized and add the pos tags and the tags are parsed into chunk trees. The nltk library has a pre-trained namied entity chunker which can be done using ne_chunk() method. The below code is an example of how to chunk the sentence.`

# In[4]:


tokens = word_tokenize(sentence1)
tags = pos_tag(tokens)
ne_tree = ne_chunk(tags)
print(ne_tree)


# `The code below is an example to count the NE but the desired output is shown in the next cell`

# In[5]:


import nltk
from collections import Counter   
for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence1))):
    if hasattr(chunk, 'label'):
        print([Counter(label) for label in chunk])


# In[6]:


#Desired Output
#ORG:1
#GPE:2
#PERSON


# `The below two cells are examples to find entities through Regex patterns`

# In[7]:


my_sent = "Google to develop AI model supporting 1,000 popular global languages Google has already integrated 1000 popular languages across the world into Google Search, despite the criticism about their functionality. The Models of language were reportedly facing some flaws, like re-enacting harmful societal biases like xenophobia, racism and more,Berchmans said on Monday that WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
word = nltk.word_tokenize(my_sent)   
pos_tag = nltk.pos_tag(word)   
chunk = nltk.ne_chunk(pos_tag)  
grammar = "NP: {<NN><NNS>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(chunk)
NE = [ " ".join(w for w, t in ele) for ele in result if isinstance(ele, nltk.Tree)]   
print (NE)


# In[8]:


grammar = "NP: {<DT><JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(chunk)
NE = [ " ".join(w for w, t in ele) for ele in result if isinstance(ele, nltk.Tree)]   
print (NE)


# In[ ]:




