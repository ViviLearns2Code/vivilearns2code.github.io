---
layout: post
title:  "Getting Started with NLP Part I"
date:   2018-11-11 10:04:00 +0100
comments: true
categories: nlp
excerpt_separator: <!--more-->
---
Two weeks ago, I attended [PyCon.DE](https://de.pycon.org/). It was an amazing experience; the talks and the people I met there have become an additional source of motivation and inspiration for me. From the people who worked as data scientists, I heard a lot about containerization or working with open source projects like Dask or Arrow. Training models in Jupyter Notebook is only a tiny part in a data scientist's daily work - handling infrastructure requirements is often more important. Below is a list of talks about topics I would like to try out in the future or read about in more detail:

* [Your First NLP Project: Peaks and Pitfalls of Unstructured Data](https://www.youtube.com/watch?v=bf2hISLgK84)
* [Germany's Next Topic Model](https://www.youtube.com/watch?v=sI7VpFNiy_I)
* [Beyond Jupyter Notebooks: Building Your Own Data Science Platform With Python & Docker](https://www.youtube.com/watch?v=xU2KUE68yFE)
* [Introduction to Docker for Pythonistas](https://www.youtube.com/watch?v=Fqq6F68SQFY)
* [Scalable Scientific Computing Using Dask](https://www.youtube.com/watch?v=OhstDq8l3OM)
* [Fulfulling Apache Arrow's Promises: Pandas On JVM Memory Without A Copy](https://www.youtube.com/watch?v=xW7IOaQvDsU)


The first talk introduced NLP library [spaCy](https://spacy.io/), which I chose to try out in my next project. Coincidentally, the Yelp company booth was advertising their [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge), and one of their available datasets contained user reviews. And so, my first NLP project was born: Sentiment analysis based on Yelp review data. 

<!--more-->

# The dataset
The Yelp reviews dataset is a 4GB file with line-delimited JSON objects. With `head <path to json file>` we can see the first couple of lines to get a feeling for the format[^1].
```javascript
{
    "review_id": "LWUtqzNthMM3vpWZIFBlPw",
    "user_id":"msQe1u7Z_XuqjGoqhB0J5g",
    "business_id":"atVh8viqTj-sqDJ35tAYVg",
    "stars":2,
    "date":"2012-11-09",
    "text":"Food is pretty good, not gonna lie. BUT you have to make sacrifices if you choose to eat there. It literally takes an hour to an hour and a half to deliver food.Seriously. EVERY SINGLE TIME. Doesnt matter if we order at 8am, 10am or 1pm. Never fails, they take F-O-R-E-V-E-R. If you dont get what you ordered or you are upset by them delivering your breakfast around LUNCH time, be ready to have the owner talk down to you and be a total bitch to you for i dont know, justwanting what you pay for?! \n\nIts over priced. But its decently tasteful food. Takes forever. Owners awitch. And i'm pretty sure that they continuing forget to pack my extra ranch just to piss me off. \n\nEnd Rant. \n\nPS- I've never gone in there to eat because i frankly, i'd rather tip the nice delivery driver then the ignorant imbeciles that work in the dining area. \n\nPPS- My hot chocolate today was cold.They should call it Cold Chocolate. Or start caring if their hot chocolate is hot. One of the two wouldbe great!",
    "useful":1,
    "funny":2,
    "cool":1
}
```
The dataset has a size too big to be comfortably loaded into memory at once with pandas. We therefore looked into [Dask](https://dask.org/), a pandas-compatible library which can handle big data. With dask, we can load and (parallelly) process partitions of big dataframes.

After experimenting around for a while, the first preprocessing stage was defined with the following steps
* replace non-English reviews with a placeholder text "NON_ENGLISH_REVIEW"[^2]
* remove URLs and email addresses from review texts 
* remove `\n`, `=`, `"`, `-`, `)`, `(` from review texts
* adjust expressions like _"Awesome!!!My"_ (becomes _"Awesome. My"_) so that spaCy won't identify the entire expression as one token later on
* map number of stars[^3] to a sentiment class in $$\{-1,0,1\}$$
* discard everything but the processed text and the sentiment class
* store processed dataset in parquet files

The above steps are applied before using spaCy. With spaCy, we can drop all stop words (important words like "not" or "no" were removed from spacy's stop word list beforehand), punctuations and non-alphabetic tokens. Here we use the recommended `nlp.pipe()`, which is internally optimized for processing sequences of texts. After processing, we throw away all reviews with less than 10 tokens, leaving a 2GB dataset with

* 5,945,754 English reviews, among which
* 3,940,130 are positive with an average length of 96.93 tokens
* 666,129 are neutral with an average length of 131.375 tokens
* 1,339,495 are negative with an average length of 140.80 tokens

Below is a distribution plot for the number of tokens in each sentiment class.
![distplot tokens][distplot]

With spaCy, we can also look at the words identified as out-of-vocabulary. These words are mostly typos or exotic food names (_"Schweinsbraten"_, _"etouffee"_, _"sopapillas"_). We replace these words with a placeholder "UNK". There is one exception to the rule: spaCy is currently unable to recognize the word "number" and we keep the word as it is.

SpaCy's part-of-speech tagger[^4] comes in very handy when it comes to categorizing words as adjectives, nouns or verbs. We use this information to analyze which adjectives are predominantly used for different sentiment classes. The word cloud at the top of the picture shows the 50 most used adjectives in positive reviews, the word cloud in the middle shows the top 50 words for neutral reviews and the word cloud at the bottom shows the top 50 words for negative reviews. We can see that there is a difference, though words like "nice", "good" or "great" appear in all sentiment classes. Unfortunately, the word clouds do not take negated adjectives like "not very good" into account.

![word cloud][cloud]

After these first insights, we can proceed with the actual sentiment analysis in the next blog post. The relevant code will be uploaded to [this repo](https://github.com/ViviLearns2Code/yelp-review).


[^1]: Yelp documents their datasets [here](https://www.yelp.com/dataset/documentation/main)
[^2]: For language identification, one possible library is [langid](https://github.com/saffsd/langid.py)
[^3]: See [Yelp review scale](https://www.yelp.com/developers/display_requirements)
[cloud]: {{"/images/adjective_cloud.png"}}
[distplot]: {{"/images/token_distribution.png"}}
[^4]: There are two token attributes `pos_` and `tag_`. The `pos_` attribute covers the [word type](http://universaldependencies.org/u/pos/) whereas the `tag_`attribute has more [detail](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)