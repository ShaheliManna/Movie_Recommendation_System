import pandas as pd #python library used for working with tabular data and is used for data analysis
import numpy as nm #python library containing mathematical functions like matrices, masked arrays etc
import difflib #python module used for comparing texts and generating reports
#required the installation of scikit-learn and used the TfidVectorizer to transform text data into matix form
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity #cosine_similarity is used to find the similarity between two vectors
movies_data = pd.read_csv("Projects/movie.csv") #read the csv file
selected_features=['Genre','Rating'] #selecting key features on the basis of which a movie shall be recommended
for feature in selected_features: #for loop to iterate through the selected features
    if feature in movies_data.columns: #Checking if the selected column exists
        '''      
        fillna() is a feature in pandas used to replace missing values with 0 by default or with a certain string if specified
        .astype(str) converts entire columns or DataFrames to strings.
        '''
        movies_data[feature] = movies_data[feature].fillna('').astype(str)
    else:
        print(f"Warning: Column '{feature}' not found in the DataFrame. Please check your CSV headers.")
combined_features=movies_data['Genre']+' '+movies_data['Rating'] #combining the selected features
vectorizer=TfidfVectorizer() #creating an instance of the TfidVectorizer
feature_vectors=vectorizer.fit_transform(combined_features) #transforming and fitting the text data into feature vectors
similarity=cosine_similarity(feature_vectors) #calculating the cosine similarity
print('Please enter the details to get the appropriate movie suggestion')
#taking inputs from the user to recommend a movie according to the appropriate key features
movie_genre=input('Enter the genre: ')
movie_rating=input('Enter the rating: ')
input_features_string = f"{movie_genre} {movie_rating}" #combining the input features
user_input_vector = vectorizer.transform([input_features_string]) #transforming the input features into feature vectors
#calculating similarity between the input vectors and the feature vectors
similarity_to_user_input = cosine_similarity(user_input_vector, feature_vectors)
#enumerate() loops through an iterable like a list and have access to both the index as well as the element itself
similarity_score_for_user_input = list(enumerate(similarity_to_user_input[0])) #How much similar to the user input
'''
sorting the movies based on the similarity score
lambda is an anonymous function and x[1] is an element of the list that is to be sorted from index 1 (from the second element)
reverse=True states that the list if sorted from the opposite also returns the same result
'''
sorted_similar_movies = sorted(similarity_score_for_user_input, key=lambda x: x[1], reverse=True)
similarity_threshold = 0.2 #setting a limit on the similarity score
print('Movie/Movies suggested for you: \n')
i=1 #starting the iterations
found_relevant_movies=False #boolean variable to check if any relevant movies are found and it retuns false
#displaying the columns from the csv dataset
display_columns = ['MovieID', 'Title', 'Director', 'ReleaseDate', 'RuntimeInHrs', 'Rating', 'Genre'] 
for movie in sorted_similar_movies: #forloop to iterate through the movies
    index, score = movie[0], movie[1] #extracting the index and the similarity score
    if score >= similarity_threshold: #checking condition if the similarity score is greater than the threshold
        #checking if the display columns are present in the csv file
        missing_cols = [col for col in display_columns if col not in movies_data.columns] 
        if missing_cols:
            #if missing we join the missing columns and print the error message
            print(f"Error: The following display columns are missing from your DataFrame: {', '.join(missing_cols)}")
            print("Please ensure your CSV file contains these columns with exact matching names.")
            break #breaks out of the loop if the error is encountered
        movie_details = movies_data.loc[index, display_columns] #selecting specific rows and columns from the csv file
        #printing the recommendation numbers and the similarity score
        print(f"--- Recommendation {i} (Similarity Score: {score:0.2f}) ---")
        #printing the details of the specific movies that are recommended
        print(f"MovieID: {movie_details['MovieID']}") #
        print(f"Title: {movie_details['Title']}")
        print(f"Director: {movie_details['Director']}")
        print(f"ReleaseDate: {movie_details['ReleaseDate']}")
        print(f"RuntimeInHrs: {movie_details['RuntimeInHrs']}")
        print(f"Rating: {movie_details['Rating']}")
        print(f"Genre: {movie_details['Genre']}")
        print("-" * 30) #printing a line to separate the recommendations
        i+=1 #increment
        found_relevant_movies=True #boolean variable to check if any relevant movies are found and it returns true
    else:
        break
if not found_relevant_movies:
    print("Sorry, no highly relevant movies found based on your criteria.")