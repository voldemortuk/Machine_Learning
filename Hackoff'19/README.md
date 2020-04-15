HackOff2019

Problem

We are trying to make a cohesive platform/application which can provide the most important points from a chapter of a book, so that no important details can be left behind unattended by the reader.

We are using a simple NLP model(Text Summarizer) for "dissecting contextual word embedding" technology. This application can be a game changer in the field of how we read and understand a text , paragraph , notes, documents, newspaper and even medical reports.

This product can act as the ultimate tool for the highly attention deficit teenagers, who cannot afford undivided attention for reading, let say -a novel or a research paper and legal terms in law.

The ultimate usefulness of this product can be visualized for college students who at the time of revision can simply upload the text and obtained the most emphasised and important concepts.

Solution

We used the gensim library in python that has prebuilt NLP ranking models to create our summaries. But this is not enough. All users do not want the same amount of summary from a given amount of text. So we added two sumamry lengths for now, 20% and 40%. But even that is not enough. If a student still has to read a couple of hundred lines, its not really a big relief.

This is where our app comes in. Our team used flutter to make the frontend of an app that has a given amount of (user defined) cards.

So whole of the book/pdf will be converted into n number of sections and then it will be presented to the user in the form of n number of cards generated and the user can happily learn through it in their own pace and timespan.

These cards hold a snippet of the summary for the user to read in their own time and pace. Considering the user splits the summary into multiple cards, this double reduction system makes sure that the user will be able to read and assimilate a given snippet, and therefore the entire summary after reading them all.
