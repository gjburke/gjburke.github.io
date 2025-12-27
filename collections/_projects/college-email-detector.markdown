---
layout: project
name: college-email-detector
description: Tool to detect college promotional emails in your inbox
link: github.com/gjburke/college-email-detector
logo: /assets/images/for-projects/college-email-detector/logo.svg
stack: 
 - Python
 - scikit-learn
 - Gmail API
 - NLTK
concepts:
 - Machine Learning
 - Data Pipeline
 - API Utilization
number: 0
---

**Update as of 12/27/2025:** This project was originally worked on and off from May 2020 to August 2022. Then I refactored it a little bit and put it on github in the early Summer of 2024.

# Overview

I built this tool because I was getting an obscene amount of college promotional emails in my inbox in highschool (thanks CollegeBoard). I was taking Andrew Ng's course on machine learning through Coursera, and I figured I had enough data from my own inbox to train something like the SVM spam classifier he had. So I read up on his pipeline and implemented it in python.

The college spam detector is a tool that connects to your gmail, reads through them, and tags any emails that it deems are college promotional emails.

**Technologies:** Python, scikit-learn, NLTK, Gmail API

**Concepts:** Machine Learning, SVM, Data Pipeline, API Utilization

# Approach

I used the Gmail API to scrape around 250 college promotional emails and 250 regular emails that I had labeled from my inbox. I used the Natural Language Toolkit (NLTK) to stem those words, and create a ranking of top words. I vectorized my emails with the representation of if the email contained those top words.

Then, with those processed emails and labels, I was able to train an SVM using scikit-learn. I experimented with different kernels and vector/dictionary sizes eventually achieving a validation accuracy of 95%.

This model was utilized in the overall application, in which the user connected their Gmail, decided the label name for college emails, and had the model scan their Gmail to label any college-adjacent emails

# Takeaways

This was essentially my first time actually applying machine learning to solve a problem of my own. 

Over the past year or so before this project, I had been learning theory and implementing some algorithms/models through the ML Coursera course with Andrew Ng. That course gave me a solid foundation of ML training and test principles, and some basic understanding of these models and how they work.

Once I finished the course, I used this project to apply at least some of what I head learned. Since I was in high school and annoyed at all these college promotional emails, that was my first thought.

I remember the first time that my model actually returned good validation results and worked on my own data. I think that feeling of satisfaction, knowing that I can actually utilize this technology to solve problems: that's what hooked me. Probably similar to an engineer see his robotic arm pick something up for the first time.

With this being the first real time I did this, I gained a lot of foundational experience with each step of model development and implementation. From collecting and preprocessing data to building out the API calls from the model results, the foundations were laid.