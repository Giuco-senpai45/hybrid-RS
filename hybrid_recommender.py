import random
import numpy as np
import pandas as pd
import re
from ContentTFIDF import ContentTFIDF
from ContentBasedRecommender import ContentBasedRecommender
import pickle
from KMeanCollaborative import KMeanCollaborative

def cleanText(readData):
    text = re.sub('[-=+#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', readData)
 
    return text

track = pd.read_csv('tracks.csv')
track = pd.DataFrame(track)
track.head()

genre = []
for i in track['artist_genre']:
    if i == '[]':
        i = 'NA'
        genre.append(i.strip())
    else:
        i = cleanText(i)
        genre.append(i.strip())
track['genre'] = genre

track = track[track['genre'] != "NA"]
track = track.reset_index()
track['track_popularity'] = track['track_popularity'] / 100 

ct = ContentTFIDF(track)
ct_tfidf = ct.calculateTFIDF()

content_recommender = ContentBasedRecommender(track, ct_tfidf)

with open('content_recommender.m5', 'wb') as f:
    pickle.dump(content_recommender, f)


kmean = KMeanCollaborative(track)
with open('kmean_recommender.m5', 'wb') as f:
    pickle.dump(kmean, f)

# recommended_features = content_recommender.recommend_features()
# recommended_genres = content_recommender.recommend_genre()
# e = content_recommender.feature_genre_intersection(track, recommended_features, recommended_genres)
# f = content_recommender.get_total_score()
