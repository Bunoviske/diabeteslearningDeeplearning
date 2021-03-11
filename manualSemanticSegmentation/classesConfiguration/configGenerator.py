import json
from random import randrange 




def generateLabelsFromDictionary(classesDictionary):

    labelsList = []

    
    for classe in classesDictionary:

        baseObject = {
        'categorie': 'void',
        'color': [
            0,
            0,
            0
        ],
        'id': 0,
        'id_categorie': 0,
        'name': 'unlabeled'
    }

        baseObject['categorie'] = classe
        baseObject['color'] = [randrange(255), randrange(255), randrange(255)]
        baseObject['id'] = classesDictionary.index(classe)
        baseObject['id_categorie'] = baseObject['id']
        baseObject['name'] = classe


        labelsList.append(baseObject)

    return labelsList






if __name__ == '__main__':

    classesDictionary = ['NaoAlimento',
    'AlimentoNaoRegistrado',
    'Arroz Integral',
    'Arroz Branco',
    'Feijão',
    'Arroz Integral e Feijao'
    ,'Arroz Branco e Feijao',
    'Arroz Carreteiro',
    'Macarrão',
    'Batata',
    'Batata Palha',
    'Batata Doce',
    'Pure de Batata',
    'Mandioca',
    'Inhame',
    'Grão de Bico',
    'Cuscuz',
    'Farofa',
    'Feijão Tropeiro',
    'Pure de Grão de Bico',
    'Lentilha',
    'Batata Baroa',
    'Lasanha',
    'Nhoque',
    'Risoto',
    'Torta Salgada',
    'Pirão',
    'Milho',
    'Maionese',
    'Creme de Milho',
    'Escondidinho de Carne',
    'Banana',
    'Manga',
    'Mamão',
    'Pêssego',
    'Abacate',
    'Maçã',
    'Pera',
    'Abacaxi',
    'Ameixa',
    'Amora',
    'Morango',
    'Melancia',
    'Cereja',
    'Uva',
    'Melão',
    'Goiaba',
    'Laranja sem Casca',
    'Salada de Frutas',
    'Pão Integral',
    'Pão Branco',
    'Pão de Queijo',
    'Bolo',
    'Carne',
    'Cozido de Carne',
    'Strognoff',
    'Linguiça',
    'Carne moída',
    'Peixe',
    'Frango (com osso)',
    'Frango (Filé sem osso)',
    'Ovo',
    'Queijo Cottage',
    'Mussarela de Búfalo',
    'Cogumelo',
    'Alface',
    'Alface com Tomate',
    'Alface com Legumes',
    'Vagem',
    'Cebola',
    'Cebola Crua',
    'Tomate Cereja',
    'Tomate',
    'Brocolis',
    'Cenoura',
    'Abobrinha',
    'Beringela',
    'Beterraba',
    'Abóbora',
    'Espinafre',
    'Couve',
    'Repolho',
    'Couve Flor',
    'Quiabo',
    'Ervilha',
    'Sopa',
    'Tofu',
    'Aspargo',
    'Pimentão',
    'Mix Vegetais',
    'Chuchu']

    labelsList = generateLabelsFromDictionary(classesDictionary)

    
    


    with open('classesConfig.json', 'w') as outfile:
        data = { }
        aux = { }

        for label in labelsList:
            data[label['categorie']] = label

        aux['labels'] = data
        json.dump(aux, outfile)    