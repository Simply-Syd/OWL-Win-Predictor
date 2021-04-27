from joblib import load
import pandas as pd

#Loads each model from the models folder.
mlb = load("models/mlb.joblib")
svc = load("models/svc.joblib")
nb = load("models/nb.joblib")
knn = load("models/knn.joblib")
lda = load("models/lda.joblib")

'''
#Dictionary if maps are added to user input
maps = {
    "1" : "Blizzard World",
    "2" : "Busan",
    "3" : "Dorado",
    "4" : "Eichenwalde",
    "5" : "Hanamura",
    "6" : "Havana",
    "7" : "Hollywood",
    "8" : "Horizon Lunar Colony",
    "9" : "Ilios",
    "10": "Junkertown",
    "11": "King's Row",
    "12": "Lijiang Tower",
    "13": "Nepal",
    "14": "Numbani",
    "15": "Oasis",
    "16": "Paris",
    "17": "Rialto",
    "18": "Route 66",
    "19": "Temple of Anubis",
    "20": "Volskaya Industries",
    "21": "Watchpoint: Gibraltar"
}
'''
#Creates a dictionary of heroes.
heroes = {
    "1": "Ana",
    "2": "Ashe",
    "3": "Baptiste",
    "4": "Bastion",
    "5": "Brigitte",
    "6": "D.Va",
    "7": "Doomfist",
    "8": "Echo",
    "9": "Genji",
    "10": "Hanzo",
    "11": "Junkrat",
    "12": "Lúcio",
    "13": "McCree",
    "14": "Mei",
    "15": "Mercy",
    "16": "Moira",
    "17": "Orisa",
    "18": "Pharah",
    "19": "Reaper",
    "20": "Reinhardt",
    "21": "Roadhog",
    "22": "Sigma",
    "23": "Soldier: 76",
    "24": "Sombra",
    "25": "Symmetra",
    "26": "Torbjörn",
    "27": "Tracer",
    "28": "Widowmaker",
    "29": "Winston",
    "30": "Wrecking Ball",
    "31": "Zarya",
    "32": "Zenyatta"
}

#Prints the dictionary for the user. 
for k,v in heroes.items():
    print(k + ": " + v)

#Prints directions for user input.
print()
print("Team 1 Composition: ")
print("Please enter the number associated with the hero. For each team, only use the hero's number once.")
print("Ex: 1 2 3 4 5 and 1 2 3 4 5 are valid team entries, but 1 2 2 3 4 5 and 1 2 3 4 5 6 are not.")

#Gets input and converts numbers to hero names for encoders.
team1 = input("Enter team 1 as numbers separated by a space (Ex: 1 2 3 4 5 6): ").split()
for i in range(len(team1)):
    team1[i] = heroes[team1[i]]

team2 = input("Enter team 2 as numbers separated by a space: ").split()
for i in range(len(team2)):
    team2[i] = heroes[team2[i]]

#Creates a new dataframe for team1 and team2 composition
data = pd.DataFrame({"team1": [team1], "team2" : [team2]})

#Reformats the DF using the MultiLabelBinarizer
data = data.join(pd.DataFrame(mlb.transform(data.pop('team1')), columns = mlb.classes_, index=data.index))
data = data.join(pd.DataFrame(mlb.transform(data.pop('team2')), columns = mlb.classes_, index=data.index), rsuffix= '_2')

#Creates a function to print information for given classifier
def pred(classifier):
    prediction = classifier.predict(data)
    confidence = classifier.predict_proba(data)
    print("\nThe predicted winner is:")
    if (prediction == "win"): 
        print("Team 1") 
        for hero in team1:
            print(hero, end = " ")
        print()
        confidence = confidence[0][1] * 100
        formatted = "{:.2f}".format(confidence)
        print("The confidence score for this prediction is " + formatted + "%.")
    else:
        print("Team 2")
        for hero in team2:
            print(hero, end = " ") 
        print()
        confidence = confidence[0][0] * 100
        formatted = "{:.2f}".format(confidence)
        print("The confidence score for this prediction is " + formatted + "%.")

    print("---------------------")



#Prints Linear SVC results seperately because the distance function
prediction = svc.predict(data)
confidence = svc.predict_proba(data)
distance = svc.decision_function(data)

print("SVC")
print("\nThe predicted winner is:")

if (prediction == "win"): 
    print("Team 1") 
    for hero in team1:
        print(hero, end = " ")
    print()
    confidence = confidence[0][1] * 100
    formatted = "{:.2f}".format(confidence)
    print("The confidence score for this prediction is " + formatted + "%.")
else:
    print("Team 2")
    for hero in team2:
        print(hero, end = " ") 
    print()
    confidence = confidence[0][0] * 100
    formatted = "{:.2f}".format(confidence)
    print("The confidence score for this prediction is " + formatted + "%.")

print("\nPoint distance from boundary: ", end = " ")
distance = distance[0]
formatted = "{:.2f}".format(distance)
print(formatted)
print("---------------------")

#Prints the remaining predictors
print("KNN")
pred(knn)

print("Naive Bayes")
pred(nb)

print("Linear Discriminant Analysis")
pred(lda)
