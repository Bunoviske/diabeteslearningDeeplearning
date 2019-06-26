#include "carbohydrate.h"
#include "watershed.hpp"

// FASTAI 
// width 466
// height 350

#define WIDTH 250
#define HEIGHT 250
#define LINE_THICKNESS 2
string IMAGES_FOLDER = "/home/bruno/openCV/semanticSegmentation/build/images";

string labels = 
"1: arroz\n\
2: feijao\n\
3: salada\n\
4: carne\n\
5: frango\n\
6: ovo\n\
7: pure de batata\n\
8: tomate\n\
9: batata\n\
10: farofa\n";


void chooseFoods(int event, int x, int y, int flags, void *objeto);        //header funcao de callback
void chooseWshedMarkers(int event, int x, int y, int flags, void *objeto); //header funcao de callback
void createGroundTruth(int index, int food);
string getFileNameWithoutExtension(string filename);
int runManualSemanticSegmentation(Mat image, string filename);
void getImages(String folder);

carboDetector carbo; //global para ser visto pela funcao de callback
Watershed wshed;

//watershed opencv global variables
Mat markerMask, img, img0, groundTruth;
Mat imgGray;
Mat markers = Mat(img.size(), CV_32S);
Mat wshedMat = Mat(img.size(), CV_8UC3);
Point prevPt(-1, -1);

int main(int argc, char **argv)
{
    // VideoCapture cap;
    // // Le imagem/video do terminal
    // if (argc > 1)
    // {
    //     cap.open(argv[1]);
    // }
    // // Se nao tiver argumentos de entrada
    // else
    // {
    //     cout << "Insira uma imagem" << endl;
    //     return -1;
    // }

    getImages(IMAGES_FOLDER);

    string mkdir = "mkdir " + IMAGES_FOLDER + "/gt";
    system(mkdir.c_str());

    //mv /home/bruno/openCV/semanticSegmentation/build/images/*"GT.png" /home/bruno/openCV/semanticSegmentation/build/images/gt
    string mv = "mv " + IMAGES_FOLDER + "/*\"GT.png\" " + IMAGES_FOLDER + "/gt";
    cout << mv << endl;
    system(mv.c_str()); 

    return 0;
}

void getImages(String folder)
{

    /*********
     *
     * Aquisicao das imagens de treino de uma pasta
     *
     *********/

    vector<string> filenames;

    glob(folder, filenames);
    cout << folder << endl;
    cout << "Qtd de imagens: " << filenames.size() << endl;

    for (size_t i = 0; i < filenames.size(); ++i)
    {
        Mat src = imread(filenames[i]);

        if (!src.data)
            cerr << "Problem loading image!!!" << endl;
        else
        {
            cout << filenames[i] << endl;
            runManualSemanticSegmentation(src, filenames[i]);
        }
    }
}

int runManualSemanticSegmentation(Mat image, string filename)
{

    resize(image, image, Size(WIDTH, HEIGHT));
    img = image.clone();

    cvtColor(img, markerMask, COLOR_BGR2GRAY);
    cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
    markerMask = Scalar::all(0);
    groundTruth = markerMask.clone();

    img0 = img.clone();
    imshow("plate", img0);
    setMouseCallback("plate", chooseWshedMarkers, NULL);

    while (1)
    {
        char c = waitKey();
        //se desejar fechar video/streaming, apertar esc
        if ((char)c == 27) //esc
        {
            return -1;
        }
        if (c == 'n')
        {
            img = image.clone();
            markers = wshed.runWatershed(&img, &markerMask, &markers, &wshedMat, imgGray);
            break;
        }
        if (c == 'w')
        {
            img = image.clone();
            markers = wshed.runWatershed(&img, &markerMask, &markers, &wshedMat, imgGray);
        }
        if (c == 'r')
        {
            img0 = image.clone();
            img = image.clone();
            markerMask = Scalar::all(0);
            imshow("plate", img0);
        }
    }

    setMouseCallback("wshed", chooseFoods, NULL);
    cout << "********************** Selecione as classes **********************" << endl;
    while (1)
    {
        char c = waitKey();
        //se desejar fechar video/streaming, apertar esc
        if ((char)c == 27) //esc
        {
            return -1;
        }
        if (c == 'n')
        {
            break;
        }
    }

    //salva imagem GT
    string filenameWithoutExtension = getFileNameWithoutExtension(filename);
    imwrite(filenameWithoutExtension + "GT.png", groundTruth);
    groundTruth = Scalar::all(0);

    return 0;
}

void chooseWshedMarkers(int event, int x, int y, int flags, void *objeto)
{
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
        return;
    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
        prevPt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN)
        prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        Point pt(x, y);
        if (prevPt.x < 0) //primeiro clique ou primeiro ponto!
            prevPt = pt;
        line(markerMask, prevPt, pt, Scalar::all(255), LINE_THICKNESS, 8, 0);
        line(img0, prevPt, pt, Scalar::all(255), LINE_THICKNESS, 8, 0);
        prevPt = pt;
        imshow("plate", img0);
    }
}

void chooseFoods(int event, int x, int y, int flags, void *objeto)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        //cout << "Posicao do mouse (" << x << ", " << y << ")    " << endl;
        int foodIndex = wshed.getFoodRegionIndex(Point(x, y)) + 1; //soma um pois essa funcao retorna o index - 1
        int food;
        cout << labels << "Digite o numero do alimento: " << flush;
        cin >> food;
        cout << endl;
        createGroundTruth(foodIndex, food);
    }
}

void createGroundTruth(int imageIndex, int food)
{
    //cout << imageIndex << endl;

    for (size_t i = 0; i < groundTruth.rows; i++)
    {
        for (size_t j = 0; j < groundTruth.cols; j++)
        {
            if (markers.at<int>(i, j) == imageIndex)
            {
                groundTruth.at<uchar>(i, j) = (uint8_t)food;
            }
        }
    }
}

string getFileNameWithoutExtension(string filename)
{
    string result = "";
    for (int i = 0; i < filename.size(); i++)
    {
        if (filename[i] == '.')
            break;
        else
            result += filename[i];
    }
    return result;
}
