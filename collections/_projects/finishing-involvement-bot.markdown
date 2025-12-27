---
layout: project
name: finishing-involvement-bot
description: Site to match UIUC students to relevant RSOs using neural networks
link: github.com/gjburke/finishing-involvement-bot
logo: /assets/images/for-projects/finishing-involvement-bot/logo.svg
stack: 
 - Python
 - Flask
 - Keras
 - SQLite
concepts:
 - Machine Learning
 - Neural Networks
 - Web Development
number: 1
---

**Update as of 12/27/2025:** I started this project in the second semester of my Freshman year, and finished it during the start of the Summer of 2024.

# Overview

The UIUC Involvement Bot is a website that recommends clubs at the University of Illinois Urbana-Champaign to students based off of their interests. Students input what type of clubs they are looking for, and the website uses two neural networks and a database of clubs to match the category and intensity level that the user wants.

This project started out with a group of around 14 people, but had to be completed on my own. With around a month left in the semester, the project manager went MIA, and the project fell dormant. I wanted to finish it because I thought it was interesting, so I took it upon myself to do it during the summer.

**Technologies:** Python, Flask, Keras, SQLite, HTML/CSS/JS

**Concepts:** Machine Learning, Neural Networks, Web Development

# Approach

This was one of the first "full applications" that I made incorporating machine learning, as small as it was. 

The whole application is driven by two neural network models: one for classifying the category of the club described by the user's request, and one for the intensity level of the club.

In order to train the models, we created a dataset of all the general club categories that we could find at UIUC. Then, as a group, we wrote prompts that would classify into each of those categories as well as "intensity levels" (how involved/committed you have to be for the club). We had over 1000 prompts in total to train around 15 categories and three intensity levels.

Once we had all of these prompts and associated category and intensity level, we vectorized the prompts so they could be fed into a neural network. This meant stemming the words, finding the most common words, and creating a vector on if those words are included or not for each prompt.

I conducted many different tests of different neural network architectures and layer types, achieving accuracies of 89.4% and 95.6% for the category and intensity level models, respectively.

In the flask backend, we vectorize the user's request and run our models on it to get the category and intensity level. From there, we can just filter and order the club database based on relevance. Those matches are then displayed to the user.

# Takeaways

This was a very formative early project and experience for me. It introduced me to backend/frontend, requests, APIs. It also was my first experience working in a software team (unfortunately not a great experience though).

Starting with the technical side, this was the first time I felt I had been on all aspects of an application. I worked on developing the model from start to finish, the frontend interactions, and the backend endpoints. It was a great experience to get some beginner experience with all of these aspects, and help me find out that I liked working in the backend and model side much more.

On a more personal side, this project really tested my resilience. I was introduced to a lot of new concepts and experiences, and had to work through and with them. Learning how to work well with my teammates and taking the lead when I needed was impactful. Having to take ownership of the project when the team lead left really made me realize I actually enjoy this stuff. And it was one of the contributing factors to me becoming a project manager the next year.

This project, although tough at times, was a great beginning to what I hope will be a life of continued learning and exploration of the aspects of building useful applications with artificial intelligence.