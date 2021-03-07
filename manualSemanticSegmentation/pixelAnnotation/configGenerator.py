import json
from random import randrange 




def generateLabelsFromDictionary(classesDictionary):

    labelsList = []

    
    for classe in classesDictionary:

        baseObject = {
        "categorie": "void",
        "color": [
            0,
            0,
            0
        ],
        "id": 0,
        "id_categorie": 0,
        "name": "unlabeled"
    }

        baseObject["categorie"] = classe
        baseObject["color"] = [randrange(255), randrange(255), randrange(255)]
        baseObject["id"] = classesDictionary.index(classe)
        baseObject["id_categorie"] = baseObject["id"]
        baseObject["name"] = classe


        labelsList.append(baseObject)

    return labelsList






if __name__ == "__main__":

    classesDictionary = ['Arroz', 'Fritas']

    labelsList = generateLabelsFromDictionary(classesDictionary)

    
    


    with open('classesConfig.json', 'w') as outfile:
        data = { }
        aux = { }

        for label in labelsList:
            data[label["categorie"]] = label

        aux['labels'] = data
        json.dump(aux, outfile)    