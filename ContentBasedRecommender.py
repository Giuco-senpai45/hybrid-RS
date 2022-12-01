import random
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))

class ContentBasedRecommender:
    
    def __init__(self, data, tfidf):
        self.data = data
        self.tfidf = tfidf
        
    def user_info(self):
        songs = list(self.data['track_name'].values)
        song = random.sample(songs, 5)

        total_dictionary = {}
        qs=[]
        qs.append("What's your favourite song out of these?   1) {}  2) {}  3) {}  4) {}  5) {}".format(song[0],song[1],song[2],song[3],song[4]))
        qs.append("Light or dark song?   1) light  2) dark")
        qs.append("Fast songs or slow songs?   1) fast  2) slow")
        qs.append("What's your current mood?   1) happy  2) meh  3) sad")
        qs.append("end")

        for q in qs:
            question = q
            if question == "end":
                break
            else:
                total_dictionary[question] = ""

        for i in total_dictionary:
            print(i)
            answer = input()
            total_dictionary[i] = answer 

        a = list(total_dictionary.items())
        self.music = song[int(a[0][1])]
        self.mood = int(a[1][1])
        self.speed = int(a[2][1])
        self.emotion = int(a[3][1])
        
        return [self.music, self.mood, self.speed, self.emotion]
    
    
    def recommend_features(self, top=200):
    
        scaler = MinMaxScaler()

        # get the index of the user's favourite track
        index = self.data[self.data['track_name'] == self.music].index.values

        # 
        track_new = self.data[['danceability', 'energy', 'valence', 'tempo', 'acousticness']]
        track_scaled = scaler.fit_transform(track_new)
        target_index = track_scaled[index]

        euclidean = []
        for value in track_scaled:
            eu = euclidean_distance(target_index, value)
            euclidean.append(eu)

        self.data['euclidean_distance'] = euclidean
#         sim_feature_index = self.data[self.data.index != index[0]].index
#         result_features = self.data.iloc[sim_feature_index].sort_values(by='euclidean_distance', ascending=True)[:top]
        result_features = self.data.sort_values(by='euclidean_distance', ascending=True)[:top]
    #     result = track.iloc[sim_feature_index][:10]

        return result_features[['id','artist_name', 'track_name', 'euclidean_distance']]

    
    def recommend_genre(self, top=200):
        
        # TF-IDF
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        tf_genre = tfidf.fit_transform(self.data.genre)

        # cosine similarity
        ts_genre = cosine_similarity(tf_genre, tf_genre)

        #Extract specific genre information
        target_genre_index = self.data[self.data['track_name'] == self.music].index.values

        # Add similarity data frame for input song
        self.data["cos_similarity"] = ts_genre[target_genre_index, :].reshape(-1,1)
        sim_genre = self.data.sort_values(by="cos_similarity", ascending=False)
        final_index = sim_genre.index.values[ : top]
        result_genre = self.data.iloc[final_index]

        return result_genre[['id','artist_name', 'track_name', 'cos_similarity']]
    
    def feature_genre_intersection(self, track, recommended_feature, recommended_genre):
        
        # Recommendation according to genre / song mood / song speed / user's mood
        intersection = pd.merge(recommended_feature, recommended_genre, how='inner')
        similarity = intersection[['euclidean_distance', 'cos_similarity']]
        scaler = MinMaxScaler()
        scale = scaler.fit_transform(similarity)
        scale = pd.DataFrame(scale, columns=['eu_scaled', 'cos_scaled'])
        
        intersection['euclidean_scaled'] = scale['eu_scaled']
        intersection['cosine_scaled'] = scale['cos_scaled']
        intersection['ratio'] = intersection['euclidean_scaled'] + (1 - intersection['cosine_scaled'])
        result_intersection = intersection.sort_values('ratio', ascending=True)
        self.result = pd.merge(track, result_intersection, how='inner').sort_values(by='ratio')
        
        return self.result

    
    def get_genre_score(self):
        cosine_sim_score = cosine_similarity(self.tfidf, self.tfidf)
        target_genre_index = self.result[self.result['track_name'] == self.music].index.values
        genre_score = cosine_sim_score[target_genre_index, :].reshape(-1, 1)
        genre_score = self.data['cos_similarity']
        return genre_score

    
    def get_mood_score(self):
        temp = pd.DataFrame(self.result['valence'])
        if self.mood == 1:
            temp['mood_score'] = temp['valence']
        else:
            temp['mood_score'] = temp['valence'].apply(lambda x: 1-x)
        return temp['mood_score']
    
    
    def get_speed_score(self):
        temp = pd.DataFrame(self.result['tempo'])
        temp['tempo_scaled'] = MinMaxScaler().fit_transform(pd.DataFrame(temp['tempo']))
        if self.speed == 1:
            temp['speed_score'] = temp['tempo_scaled']
        else:
            temp['speed_score'] = temp['tempo_scaled'].apply(lambda x: 1-x)
        return temp['speed_score']

    
    def get_emotion_score(self):
        temp = self.result[['danceability', 'energy', 'acousticness']]
        temp['danceability_scaled'] = MinMaxScaler().fit_transform((pd.DataFrame(temp['danceability'])))
        temp['acousticness_reverse'] = temp['acousticness'].apply(lambda x: 1-x)
        if self.emotion == 1:
            temp['emotion_score'] = temp.apply(lambda x: 1/3 * (x['danceability_scaled'] + x['energy'] + x['acousticness_reverse']), axis = 1)
        elif self.emotion == 2:
            temp['emotion_score'] = temp.apply(lambda x: 2/3 * (abs(x['danceability_scaled']-0.5) + abs(x['energy']-0.5) + abs(x['acousticness_reverse']-0.5)), axis = 1)
        else:
            temp['emotion_score'] = temp.apply(lambda x: 1/3 * ((1-x['danceability_scaled']) + (1-x['energy']) + (1-x['acousticness_reverse'])), axis = 1)
        return temp['emotion_score']

    def get_total_score(self, top_n = 10):
        result_df = self.result[['artist_name', 'track_name', 'album_name']]
        result_df['mood_score'] = pd.DataFrame(self.get_mood_score())
        result_df['speed_score'] = pd.DataFrame(self.get_speed_score())
        result_df['emotion_score'] = pd.DataFrame(self.get_emotion_score())
        result_df['genre_score'] = pd.DataFrame(self.get_genre_score())
        result_df['total_score'] = result_df.apply(lambda x: 1/6*(x['mood_score'] + x['speed_score'] + x['emotion_score']) + 0.5*x['genre_score'], axis = 1)
        
        return result_df.iloc[1:].sort_values(by ='total_score', ascending=False)[:top_n]
    