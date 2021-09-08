import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
movies = pd.read_csv('movies_metadata.csv')

tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
overview_matrix = tfidf.fit_transform(movies['overview'])
similarity_matrix = linear_kernel(overview_matrix,overview_matrix)
mapping = pd.Series(movies.index,index = movies['title'])

def recommend_movies(movie_input):
    if movie_input not in mapping:
        print("enter the valid movie name")
        return

    movie_index = mapping[movie_input]
    similarity_score = list(enumerate(similarity_matrix[movie_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:50]
    movie_indices = [i[0] for i in similarity_score]
    rd.shuffle(movie_indices)
    l=[movie_indices[i] for i in range(0,10)]
    return (movies['title'].iloc[l])

def random_recommend_movies(movie_input):
    movie_index = mapping[movie_input]
    similarity_score = list(enumerate(similarity_matrix[movie_index]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:950]
    movie_indices = [i[0] for i in similarity_score]
    rd.shuffle(movie_indices)
    l=[movie_indices[i] for i in range(0,10)]
    return (movies['title'].iloc[l])

def top_rated():
    mapping_1 = pd.Series(movies['vote_average'],index = movies.index)
    mapping_2 = pd.Series(movies['title'],index = movies.index)
    a = mapping_1.sort_values(axis='index',ascending=False)
    c=0
    for i in a.index:
        if(c<10):
            for j in mapping_2.index:
                if(i==j):
                    print(mapping_2[j],a[i])
                    c+=1
                    break
        else:
            break

while(1):
    print("################################################################")
    print('1.randomly any movie')
    print('2.Related to any movie you want')
    print('3.top 10 rated movies')
    print('4.exit')
    ch=int(input('Enter your choice = '))
    if(ch==1):
        print(random_recommend_movies(rd.choice(mapping)))
    elif(ch==2):
        name=input("enter your favourite movie:")
        print(recommend_movies(name))
    elif(ch==3):
        top_rated()
    elif(ch==4):
        break
    else:
        print('enter the valid choice')
    print("################################################################")