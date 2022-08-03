// OpenCVApplication.cpp : Defines the entry point for the console application.
//
//Birlutiu Claudiu-Andrei, gr 30236, Proiect Kmeans
#include "stdafx.h"
#include "common.h"
#include<limits>
#include <time.h>
using namespace std;
using namespace cv;

#define MAX_VALUE 255
#define MAX_STEPS 10

/* Aceasta metoda va calcula histograma si numarul de intensitati prezente in imagine 
*  Numarul de intensitati prezente va fi folosit ca verificare in cazul in care utilizatorul da o valoare mai mare pentru cluster
*/
void computeHistogram(Mat modul,int* histogram, int n, int *numberOfIntensities) {
	int height = modul.rows;
	int width = modul.cols; 
	*numberOfIntensities = 0;
	for (int i = 0; i < n; i++) {
		histogram[i] = 0;  
	}
	for (int i = 1; i < height - 1; i++){
		for (int j = 1; j < width - 1; j++){
			if (!histogram[modul.at<uchar>(i, j)])
				(*numberOfIntensities)++;
			histogram[modul.at<uchar>(i, j)] += 1;
		}
	}
}

//functie pentru citirrea unei imagini
void openImage(Mat* src, int option) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	(*src) = imread(fname, option);
}

/////////////////////////////////////////////////// K MEANS  ------ GRAYSCALE  //////////////////////////////////////////////////

typedef struct {
	int mean;
	int computingMean;
	int currentNbOfPoints;   //numarul de puncte curente ce intra in calcularea computingMean
}CENTROID;
//citeste de la consola numarul de clustere si se verifica cu limita posibila data de histograma
int getFromConsoleNbOfCluster(int limit) {
	int k = 0;
	for (;;) {
		printf("Write the number of cluster: k=");
		if (scanf("%d", &k) != 1 || k<1) {
			printf("Write a positive number\n");
			continue;
		}
		if (k >= limit) {
			printf("The possible clusters for this image are %d. There are generated so %d clusters", limit, limit);
			return limit;
		}
		return k;
	}
}
void selectRandomCentroids(Mat src, int *histogram , vector<CENTROID>* centroids) {
	int numeberOfIntensities;
	//se construieste histograma imaginii
	computeHistogram(src, histogram, 256, &numeberOfIntensities);
	//se ia un vector ce afirma faptul ca locul respectiv a fost vizitat si se ignora
	int k = getFromConsoleNbOfCluster(numeberOfIntensities);  //numarul e limitat la valorile posibile din histograma
	int visited[256] = { 0 };
	srand(time(NULL));
	int nbOfClusters = 0;
	while (nbOfClusters != k) {
		//se ia un numar random din cele 256 posibile 
		int intensity = rand() % 256;
		//daca clusterul cu media aceasta nu a fost adaugat inainte si exista intensitatea in imagine, atunci se
		//va adauga noua valoare in cluster
		if (!visited[intensity] && histogram[intensity]) {
			centroids->push_back(CENTROID{ intensity, 0, 0 });
			nbOfClusters++;
			visited[intensity] = 1;
		}
	}
	printf("Initial centroids:\n ");
	for (int k = 0; k < centroids->size(); k++) {
		printf("Centroid[%d] -> %d\n", k, (*centroids)[k].mean);
	}
}
void KmeansGrayscale(Mat src, Mat *dst) {
	vector<CENTROID> centroids;   //acest vector va contine valoarea veche a clusteru;ui, valoarea medie la un moment dat si 
							   //numarul de elemente curente din cluster la rularea o iteratie noua
	//se calculeaza histrograma pentru a vedea cate valori diferite sunt pentru intensitate si tot cu ajutorul acesteia se vor determinavalorile initiale pentru clustere
	int histogram[256];
	selectRandomCentroids(src, histogram, &centroids);
	int belongClusterVec[256] = { 0 };   //vector de apartenenta cluster
	bool convergenta = false;   //nicio schimbare la nivel de clustere
	//intrare bucla iterativa
	for (int step = 0; step < MAX_STEPS && !convergenta; step++) {
		convergenta = true;
		//parcurgere histograma si redistribuire valori
		for (int intensity = 0; intensity < 256; intensity++) {
			if (histogram[intensity] == 0)
				continue;   //se trece la urmatoarea intensitate daca nu exista in imagine
			//se ia fiecare cluster si se verifica care e cel mai apropia de valoarea curenta
			int valMin = 256; //se initializeaza valoarea minima 
			int clusterIndex = 0;        //index-ul clusterului cel mai apropiat
			for (int k = 0; k < centroids.size(); k++) {
				if (abs(intensity - centroids[k].mean) < valMin) {    //se calculeaza distanta dintre valoarea centrooidului si a punctului curent
					valMin = abs(intensity - centroids[k].mean);
					clusterIndex = k;
				}
			}
			//actualizare cluster
			belongClusterVec[intensity] = clusterIndex;  //actualizare apartenenta cluster pentru valorile intensitatiilor
			//calculare noua medie
			int newCurrentPoints = centroids[clusterIndex].currentNbOfPoints + histogram[intensity];
			int newCurrentMean = (centroids[clusterIndex].currentNbOfPoints * centroids[clusterIndex].computingMean +
				histogram[intensity] * intensity) / newCurrentPoints;
			centroids[clusterIndex].computingMean = newCurrentMean;
			centroids[clusterIndex].currentNbOfPoints = newCurrentPoints;
		}

		//actualizare centroide pentru noua iteratie
		for (int k = 0; k < centroids.size(); k++) {
			if (centroids[k].mean != centroids[k].computingMean)
				convergenta = false;  // in cazul in care s-a schimbat valoasrea medie a unui cluster se va executa o noua iteratie
			centroids[k].mean = centroids[k].computingMean;
			centroids[k].computingMean = 0;
			centroids[k].currentNbOfPoints= 0;
		}
	}

	*dst = Mat::zeros(src.size(), CV_8UC1);  //initializare matrice destinatie
	//parcurgere imagine sursa si actualizare destinatie cu valoarea clusterului
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int indexClusterAtasat = belongClusterVec[src.at<uchar>(i, j)];
			dst->at<uchar>(i, j) = centroids[indexClusterAtasat].mean;
		}
	}
	printf("Convergenta: %d\n", convergenta);
	//afisare centroide 
	printf("Final centroids:\n ");
	for (int k = 0; k < centroids.size(); k++) {
		printf("Centroid[%d] -> %d\n",k, centroids[k].mean);
	}
}
//aceatsa functie este apelata din main; se va citi o imagine grayscale din memorie, se va aplica algortimul Kmeans si se va afisa imaginea rezultata
void KMeansGray() {
	Mat src;
	Mat dst;
	openImage(&src, CV_LOAD_IMAGE_GRAYSCALE);
	KmeansGrayscale(src, &dst);
	imshow("Sursa", src);
	imshow("Destinatie", dst);
	waitKey(0);
}

///////////////////////////////////////////////////// K MEANS -------- COLOR - pe BGR - nu da rezultate vizibile din cauza iluminarii  //////////////////////////////////////////
typedef struct {
	int x; 
	int y; 
	int z;
}Point3;
typedef struct {
	Point3 mean;              //este un punct (b,g,r) 
	Point3 computingMean;     //este un punct (b,g,r) 
	int currentNbOfPoints;   //numarul de puncte curente ce intra in calcularea computingMean-ului
}CENTROID_ab;

//aceasta functie va calcula histograma 3d pentru o o imagine color BGR
void compoute3DHistogram(int**** histogram, Mat Lab, int *nbOfColor) {
	(*histogram) = (int***)malloc(256 * sizeof(int**));
	if (!(*histogram)) {
		fprintf(stderr, "Could not alloc memmory for 3d histogram");
		exit(1);
	}
	for (int i = 0; i < 256; i++) {
		(*histogram)[i] = (int**)malloc(256 * sizeof(int*));
		if (!(*histogram)[i]) {
			fprintf(stderr, "Could not alloc memmory for 2d histogram");
			free(*histogram);
			exit(1);
		}
		for (int j = 0; j < 256; j++) {
			(*histogram)[i][j] = (int*)calloc(256, sizeof(int));
		}
	}
	*nbOfColor = 0;
	for (int i = 0; i < Lab.rows; i++) {
		for (int j = 0; j < Lab.cols; j++) {
			uchar b = Lab.at<Vec3b>(i, j)[0];
			uchar g = Lab.at<Vec3b>(i, j)[1];
			uchar r = Lab.at<Vec3b>(i, j)[2];
			if ((*histogram)[b][g][r]== 0) {
				(*nbOfColor)++;
			}
			(*histogram)[b][g][r]++;
		}
	}
}
int getIndexMinimum(int* vector, int k) {
	int minValue = vector[0];
	int min = 0;
	for (int i = 1; i < k; i++) {
		if (vector[i] < minValue) {
			minValue = vector[i];
			min = i;
		}
	}
	return min;
}
void selectCentroidsColor(int ****histogram, Mat src , vector<CENTROID_ab>* centroids) {
	//se va aloca dinamic histograma 3D
	
	int nbOfColors;   //numarul maxim de clustere posibile
	compoute3DHistogram(histogram, src, &nbOfColors);
	int k = getFromConsoleNbOfCluster(nbOfColors);   //se citesc de la tastatura numarul de clustere; valoarea maxima va fi data de numarul de culori existente in imagine
	
	//se vor lua cele mai pronuntate k culori ale histogramei
	for (int i = 0; i < k; i++) {
		centroids->push_back(CENTROID_ab{ Point3{0,0,0}, Point3{0,0,0}, 0 });
	}
	int* current = (int*)calloc(k, sizeof(int));
	for (int b = 0; b < 256; b++) {
		for (int g = 0; g < 256; g++) {
			for (int r = 0; r < 256; r++) {
				int minClusterIndex = getIndexMinimum(current, k);
				if ((*histogram)[b][g][r] > current[minClusterIndex]) {
					(*centroids)[minClusterIndex] = CENTROID_ab{ Point3{b,g,r}, Point3{0,0,0}, 0 };
					current[minClusterIndex] = (*histogram)[b][g][r];
				}
			}
		}
	}
	
	printf("Initial centroids:\n ");
	for (int k = 0; k < centroids->size(); k++) {
		printf("Centroid[%d] -> (%d %d %d)\n", k, (*centroids)[k].mean.x, (*centroids)[k].mean.y, (*centroids)[k].mean.z);
	}
	free(current);
}

float getEuclideanDistance(Point3 A, Point3 B) {
	
	return sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2) + pow(A.z - B.z, 2));
}

void KmeansColor(Mat src, Mat *dst) {
	int ***histogram;    //histograma bidimensionala; se va calcula pentru ab; se va aloca spatiu dinamic
	vector<CENTROID_ab> centroids;    //declarare vector de centroizi
	selectCentroidsColor(&histogram, src, &centroids);

	//creare matrice de apartenenta cluster; 
	int*** belongClusterMat = (int***)malloc(256 * sizeof(int**));
	if (!belongClusterMat) {
		fprintf(stderr, "Could not alloc memmory");
		exit(1);
	}
	for (int i = 0; i < 256; i++) {
		belongClusterMat[i] = (int**)calloc(256, sizeof(int*));
		if (!belongClusterMat[i]) {
			fprintf(stderr, "Could not alloc memmory");
			free(belongClusterMat);
			exit(1);
		}
		for (int j = 0; j < 256; j++) {
			belongClusterMat[i][j] = (int*)calloc(256, sizeof(int));
		}
	}
	// start iteration
	bool convergenta = false;   //nicio schimbare la nivel de clustere
	//intrare bucla iterativa
	for (int step = 0; step < MAX_STEPS && !convergenta; step++) {
		convergenta = true;    //convergenta se pune pe true

		//se parcurg punctele din imagine si se redistribuie in clustere
		for (int b = 0; b < 256; b++) {
			for (int g = 0; g < 256; g++) {
				for (int r = 0; r < 256; r++) {
					if (!histogram[b][g][r]) continue; //daca nu e aceasta valoare in imagine, se continua iteratia

					float minDistance = std::numeric_limits<float>::infinity(); //se initializeaza distanta minima la valoarea maxima 
					int clusterIndex = 0;
					for (int k = 0; k < centroids.size(); k++) {
						int distance = getEuclideanDistance(Point3{ b, g,r }, centroids[k].mean);
						if (distance < minDistance) {
							minDistance = distance;
							clusterIndex = k;
						}
					}

					//actualizare punct la cluster-ul nou
					if (belongClusterMat[b][g][r] != clusterIndex)
						convergenta = false;
					belongClusterMat[b][g][r]= clusterIndex;
					int newCurrentPoints = centroids[clusterIndex].currentNbOfPoints + histogram[b][g][r];
					int newCurrentMeanB = (centroids[clusterIndex].currentNbOfPoints * centroids[clusterIndex].computingMean.x +
						histogram[b][g][r] * b) / newCurrentPoints;
					int newCurrentMeanG = (centroids[clusterIndex].currentNbOfPoints * centroids[clusterIndex].computingMean.y +
						histogram[b][g][r] * g) / newCurrentPoints;
					int newCurrentMeanR = (centroids[clusterIndex].currentNbOfPoints * centroids[clusterIndex].computingMean.z +
						histogram[b][g][r] * r) / newCurrentPoints;
					centroids[clusterIndex].computingMean = Point3{ newCurrentMeanB, newCurrentMeanG, newCurrentMeanR };
					centroids[clusterIndex].currentNbOfPoints = newCurrentPoints;
				}
			}
		}
		//actualizare centroide pentru noua iteratie
		for (int k = 0; k < centroids.size(); k++) {
			centroids[k].mean = centroids[k].computingMean;
			centroids[k].computingMean = Point3{ 0, 0, 0 };
			centroids[k].currentNbOfPoints = 0;
		}
		printf("Step: %d\n", step);
	}
	*dst = src.clone();
	//actualizare lab dupa clustere]
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int b = src.at<Vec3b>(i,j)[0];
			int g = src.at<Vec3b>(i, j)[1];
			int r = src.at<Vec3b>(i, j)[2];
			int clusterIndex = belongClusterMat[b][g][r];
			dst->at<Vec3b>(i, j)[0] = centroids[clusterIndex].mean.x;
			dst->at<Vec3b>(i, j)[1] = centroids[clusterIndex].mean.y;
			dst->at<Vec3b>(i, j)[2] = centroids[clusterIndex].mean.z;

		}
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			free(histogram[i][j]);
			free(belongClusterMat[i][j]);
		}
		free(histogram[i]);
		free(belongClusterMat[i]);
	}
	free(histogram);
	free(belongClusterMat);
}

//////////////////////KMEANS COLOR -  parcurgere pe matricea de imagine, nu histograma
void KmeansColor2(Mat src, Mat* dst) {
	int*** histogram;    //histograma bidimensionala; se va calcula pentru ab; se va aloca spatiu dinamic
	vector<CENTROID_ab> centroids;    //declarare vector de centroizi
	selectCentroidsColor(&histogram, src, &centroids);

	//creare matrice de apartenenta cluster; 
	int*** belongClusterMat = (int***)malloc(256 * sizeof(int**));
	if (!belongClusterMat) {
		fprintf(stderr, "Could not alloc memmory");
		exit(1);
	}
	for (int i = 0; i < 256; i++) {
		belongClusterMat[i] = (int**)calloc(256, sizeof(int*));
		if (!belongClusterMat[i]) {
			fprintf(stderr, "Could not alloc memmory");
			free(belongClusterMat);
			exit(1);
		}
		for (int j = 0; j < 256; j++) {
			belongClusterMat[i][j] = (int*)calloc(256, sizeof(int));
		}
	}
	// start iteration
	bool convergenta = false;   //nicio schimbare la nivel de clustere
	//intrare bucla iterativa
	for (int step = 0; step < MAX_STEPS && !convergenta; step++) {
		convergenta = true;    //convergenta se pune pe true

		//se parcurg punctele din imagine si se redistribuie in clustere
		for(int i=0; i<src.rows; i++){
			for(int j=0; j<src.cols; j++){
				uchar b = src.at<Vec3b>(i, j)[0];
				uchar g = src.at<Vec3b>(i, j)[1];
				uchar r = src.at<Vec3b>(i, j)[2];


				float minDistance = std::numeric_limits<float>::infinity(); //se initializeaza distanta minima la valoarea maxima 
				int clusterIndex = 0;
				for (int k = 0; k < centroids.size(); k++) {
					int distance = getEuclideanDistance(Point3{ b, g,r }, centroids[k].mean);
					if (distance < minDistance) {
						minDistance = distance;
						clusterIndex = k;
					}
				}

				//actualizare punct la cluster-ul nou
				if (belongClusterMat[b][g][r] != clusterIndex)
					convergenta = false;
				belongClusterMat[b][g][r] = clusterIndex;
				int newCurrentPoints = centroids[clusterIndex].currentNbOfPoints + 1;
				int newCurrentMeanB = (centroids[clusterIndex].currentNbOfPoints * centroids[clusterIndex].computingMean.x + b) / newCurrentPoints;
				int newCurrentMeanG = (centroids[clusterIndex].currentNbOfPoints * centroids[clusterIndex].computingMean.y + g) / newCurrentPoints;
				int newCurrentMeanR = (centroids[clusterIndex].currentNbOfPoints * centroids[clusterIndex].computingMean.z + r) / newCurrentPoints;
			
				centroids[clusterIndex].computingMean = Point3{ newCurrentMeanB, newCurrentMeanG, newCurrentMeanR };
				centroids[clusterIndex].currentNbOfPoints = newCurrentPoints;
			}
		}
		
		//actualizare centroide pentru noua iteratie
		for (int k = 0; k < centroids.size(); k++) {
			centroids[k].mean = centroids[k].computingMean;
			centroids[k].computingMean = Point3{ 0, 0, 0 };
			centroids[k].currentNbOfPoints = 0;
		}
		printf("Step: %d\n", step);
	}
	*dst = src.clone();
	//actualizare destinatie dupa clustere
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int b = src.at<Vec3b>(i, j)[0];
			int g = src.at<Vec3b>(i, j)[1];
			int r = src.at<Vec3b>(i, j)[2];
			int clusterIndex = belongClusterMat[b][g][r];
			dst->at<Vec3b>(i, j)[0] = centroids[clusterIndex].mean.x;
			dst->at<Vec3b>(i, j)[1] = centroids[clusterIndex].mean.y;
			dst->at<Vec3b>(i, j)[2] = centroids[clusterIndex].mean.z;

		}
	}
	//eliberare memroie heap 
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			free(histogram[i][j]);
			free(belongClusterMat[i][j]);
		}
		free(histogram[i]);
		free(belongClusterMat[i]);
	}
	free(histogram);
	free(belongClusterMat);
}



//cu pracurgerea histogramei 
void testKMeansColor() {
	Mat src;
	Mat dst;
	openImage(&src, CV_LOAD_IMAGE_COLOR);
	KmeansColor(src, &dst);
	imshow("Sursa", src);
	imshow("Destinatie", 10*dst);
	waitKey(0);
	
}
//cu pracuregrea imaginii 
void testKMeansColor2() {
	Mat src;
	Mat dst;
	openImage(&src, CV_LOAD_IMAGE_COLOR);
	KmeansColor2(src, &dst);
	imshow("Sursa", src);
	imshow("Destinatie", 20* dst);
	waitKey(0);
}


///////////////////////////////////////////////////////////////KMEANS COLOR HSV - pe canalul H
//se face split-ul pe cele 3 canale si se apeleaza KMeans de grayscale pentru canalul H
void KmeansColorHSV(Mat src, Mat* dst) {
	Mat HSV;
	cvtColor(src, HSV, COLOR_BGR2HSV);
	Mat channels[3];
	split(HSV, channels);

	Mat newH;
	KmeansGrayscale(channels[0], &newH);
	imshow("Sursa", src);
	imshow("H nemodificat", channels[0]);
	imshow("NewH", newH);
	waitKey(0);
}

//cu pracuregrea imaginii 
void testKMeansColorHSV() {
	Mat src;
	Mat dst;
	openImage(&src, CV_LOAD_IMAGE_COLOR);
	KmeansColorHSV(src, &dst);
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Kmeans Gray\n");
		printf(" 2 - Kmeans BGR cu parcurgere histograma\n");
		printf(" 3 - Kmeans BGR cu parcurgere imagine\n");
		printf(" 4 - Kmeans BGR cu HSV pe canalul H\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			KMeansGray();
			break;
		case 2:
			testKMeansColor();
			break;
		case 3:
			testKMeansColor2();
			break;
		case 4: 
			testKMeansColorHSV();
			break;

		}
	} while (op != 0);
	return 0;
}
