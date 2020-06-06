# Bitmaps: Binarized transformers for making fast predictions

In this paper we introduce a method for reducing the memory usage of BERT by representing 32-bit floating point weights as 1-bit quantities; a process called binarization. Our approach leads to a decrease
in memory usage by a factor of up to 30, but at the cost of decreased performance. We evaluate our binarization approach for BERT on the masked language modeling pre-training objective as well as the downstream task of binary sentiment analysis on SST-2.
This project was done as part of the Natural Language Processing Capstone (CSE481N) taught by Noah Smith in Spring 2020 at the University of Washington, Seattle. All code as well as a series of blog posts describing the progression of the project can be found at https://github.com/nlp-capstone.
A presentation of our project can be found at https://vimeo.com/426452840.
