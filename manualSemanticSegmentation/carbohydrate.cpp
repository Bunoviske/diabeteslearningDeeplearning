#include "carbohydrate.h"

carboDetector::carboDetector(){
    /*********
     *
     * Definicao das relacoes de carboidrato. Foi usado estrutura map (dicionario)
     *
     *********/

    //TABELA USDA: 1 cup tem 236,588 ml!!!!!

    //relacao de CHO e densidade mapeados pelo nome

    //carboidratos em geral
    foods["arroz"] = std::make_pair(0.165, 0.73);
    foods["feijao"] = std::make_pair(0.165, 0.70); //verificar ainda efeito do feijao com caldo na densidade
    foods["arroz&feijao"] = std::make_pair(0.165, 0.75);

    foods["pasta"] = std::make_pair(0.26, 0.55);
    foods["batata"] = std::make_pair(0.2, 0.59);
    foods["batataPalha"] = std::make_pair(0.5, 0.152);
    foods["pureBatata"] = std::make_pair(0.165, 1.048);
    foods["batataDoce"] = std::make_pair(0.23, 0.65);
    foods["mandioca"] = std::make_pair(0.25, 0.63);
    foods["inhame"] = std::make_pair(0.27, 0.79);
    foods["graoDeBico"] = std::make_pair(0.27, 0.69);
    foods["cuscuz"] = std::make_pair(0.40, 0.66);
    foods["farofa"] = std::make_pair(0.40, 0.47);
    foods["milho"] = std::make_pair(0.25, 0.7);

    foods["pureGraoDeBico"] = std::make_pair(0.22, 1.048);
    //calculo empirico: foi usada densidade do pure de batata e foi diminuido a qtd
    //de carboidrato baseado na proporcao da batata ---> pure de batata

    foods["lentilha"] = std::make_pair(0.22, 0.83);
    foods["baroa"] = std::make_pair(0.25, 1); //densidade incerta
    foods["lasanha"] = std::make_pair(0.12, 1.04);
    foods["nhoque"] = std::make_pair(0.2, 0.67);
    foods["risoto"] = std::make_pair(0.24, 0.76);
    foods["tortaSalgada"] = std::make_pair(0.15, 0.60); //densidade incerta

    //frutas
    foods["banana"] = std::make_pair(0.23, 0.634);

    //paes e doces
    foods["pao"] = std::make_pair(0.5, 0.29); // ou 0.42 para pao branco!
    foods["bolo"] = std::make_pair(0.5, 0.415);

    //proteina
    foods["carne"] = std::make_pair(0.0, 0.96);
    foods["strognoff"] = std::make_pair(0.0, 0.945);
    foods["linguica"] = std::make_pair(0.0, 0.93);
    foods["carneMoida"] = std::make_pair(0.0, 0.95); //densidade incerta
    foods["peixe"] = std::make_pair(0.0, 0.96);
    foods["frangoComOsso"] = std::make_pair(0.0, 0.96);
    foods["fileFrango"] = std::make_pair(0.0, 0.59);
    foods["ovo"] = std::make_pair(0.0, 0.6);
    foods["queijoCottage"] = std::make_pair(0.0, 1);
    foods["mussarelaBufalo"] = std::make_pair(0.0, 0.946);
    foods["cogumelo"] = std::make_pair(0.0, 0.61);


    //legumes e verduras
    foods["salada"] = std::make_pair(0.0, 0.152);
    foods["salada&tomate"] = std::make_pair(0.0, 0.4); //densidade CHUTADA. fez-se a media de alface e tomate!
    foods["salada&legume"] = std::make_pair(0.0, 0.45);
    foods["vagem"] = std::make_pair(0.0, 0.676);
    foods["cebola"] = std::make_pair(0.0, 0.887);
    foods["tomateCereja"] = std::make_pair(0.0, 0.63);
    foods["tomate"] = std::make_pair(0.0, 0.76);
    foods["brocolis"] = std::make_pair(0.0, 0.65);
    foods["cenoura"] = std::make_pair(0.0, 0.54);
    foods["abobrinha"] = std::make_pair(0.0, 0.76);
    foods["beringela"] = std::make_pair(0.0, 0.419);
    foods["beterraba"] = std::make_pair(0.0, 0.72);
    foods["abobora"] = std::make_pair(0.0, 1); //densidade de abobora amassada no cup
    foods["espinafre"] = std::make_pair(0.0, 0.76);
    foods["couve"] = std::make_pair(0.0, 0.50);
    foods["repolho"] = std::make_pair(0.0, 0.635);
    foods["couveFlor"] = std::make_pair(0.0, 0.525);
    foods["quiabo"] = std::make_pair(0.0, 0.677);
    foods["ervilha"] = std::make_pair(0.0, 0.67);
    foods["sopa"] = std::make_pair(0.06, 1);
    foods["tofu"] = std::make_pair(0.0, 0.945);
    foods["aspargo"] = std::make_pair(0.0, 0.76);


    totalCarbo = 0;
    foodFeatures.clear();


    // cout << "Clique em cada regiao que representa uma comida e indique qual é o alimento" << endl;
    // cout << "Escreva conforme as opcoes listadas a seguir:" << endl;
    // for(auto it = foods.cbegin(); it != foods.cend(); ++it)
    // {
    //     std::cout << it->first << endl;
    // }
    // cout << endl;


}

void carboDetector::saveRegionPixeis(int regionPixeis, string name){

    auto it = foods.find(name);
    if(it == foods.end())
        cout << "Nome do alimento invalido" << endl;
    else{
        foodNames.push_back(name);
        foodFeatures.push_back(foodRegion(regionPixeis,it->second.first,it->second.second));
    }

}

float carboDetector::calculateCarbo(){

    saveRegionFeaturesFile();

    float aux = 0;
    for(int i = 0; i < foodFeatures.size();i++){
        aux += (foodFeatures[i].density * foodFeatures[i].regionPixeis);
    }
    constant = totalWeigh/aux;
    //cout << constant << ' ' << foodFeatures.size() << endl;

    for(int i = 0; i < foodFeatures.size();i++){
        foodFeatures[i].weigh = foodFeatures[i].density * foodFeatures[i].regionPixeis * constant;
        foodFeatures[i].carbo = foodFeatures[i].weigh * foodFeatures[i].relation;
        cout << foodNames[i] << "  Peso: " << foodFeatures[i].weigh << " Carbo: " << foodFeatures[i].carbo << endl;
        totalCarbo += foodFeatures[i].carbo;
    }

    return totalCarbo;

}

void carboDetector::saveRegionFeaturesFile(){
    //salva o numero de pixeis da regiao e seu respectivo nome para evitar a segmentacao das regioes com o mouse novamente

    fstream file;
    file.open("fileOutput.txt", ios_base::out);

    for(int i = 0; i < foodFeatures.size(); i++){
        file << foodNames[i] << ',' << foodFeatures[i].regionPixeis << endl;
    }

    file.close();


}












