#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>

#include "maxflow/graph.h"

#include "image.h"

using namespace std;
using namespace cv;

typedef pair<int,int> ii;
typedef pair<ii, ii> pp;

Mat get_gradient(Image<Vec3b> (&I)){
	Mat I_gray;
	cvtColor(I, I_gray, COLOR_BGR2GRAY);
	Mat Gx = Mat::zeros(I.rows, I.cols, CV_16SC1);
	Mat Gy = Mat::zeros(I.rows, I.cols, CV_16SC1);
	Mat G = Mat::zeros(I.rows, I.cols, CV_32FC1);

	int GxKernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
	int GyKernel[3][3] = {{-1, -2 ,-1}, {0, 0, 0}, {1, 2, 1}};

	for(int i=0;i<I_gray.rows - 3;i++){
		for(int j=0;j<I_gray.cols - 3;j++){
			Mat window_imageX(I_gray, Rect(j, i, 3, 3));
			Mat window_imageY(I_gray, Rect(j, i, 3, 3));

			for(int k=0;k<window_imageX.rows;k++){
				for(int m=0;m<window_imageX.cols;m++){
					window_imageX.at<int>(k,m) *= GxKernel[k][m];
					window_imageY.at<int>(k,m) *= GyKernel[k][m];
				}
			}

			for(int k=0;k<window_imageX.rows;k++){
				for(int m=0;m<window_imageX.cols;m++){
					if(k == 1 && m == 1) continue;

					window_imageX.at<int>(1,1) += window_imageX.at<int>(k,m);
					window_imageY.at<int>(1,1) += window_imageY.at<int>(k,m);
				}
			}

			Gx.at<int>(i+1,j+1) = window_imageX.at<int>(1,1);
			Gy.at<int>(i+1,j+1) = window_imageY.at<int>(1,1);
			G.at<float>(i+1,j+1) = hypot(Gx.at<int>(i+1,j+1), Gy.at<int>(i+1,j+1));
		}
	}

	return G;
}

float g(const Mat& G, int i, int j){
	float alpha = 1.0, beta = 1.0;
	return alpha/(1.0 + beta*(G.at<float>(i, j))*(G.at<float>(i, j)));
}

int main() {
	//testGCuts();

	Image<Vec3b> Icolor= Image<Vec3b>(imread("../fishes.jpg"));
	Mat I_gray;
	cvtColor(Icolor, I_gray, COLOR_BGR2GRAY);

	Mat G = get_gradient(Icolor);

	Image<Vec3b> blueIntensities = Image<Vec3b>(imread("../fishes.jpg"));
	Image<Vec3b> whiteIntensities = Image<Vec3b>(imread("../fishes.jpg"));
	Image<Vec3b> ge = Image<Vec3b>(imread("../fishes.jpg"));
	Image<Vec3b> gi = Image<Vec3b>(imread("../fishes.jpg"));

	for(int i=0;i<Icolor.rows;i++){
		for(int j=0;j<Icolor.cols;j++){
			whiteIntensities.at<Vec3b>(i,j).val[0] = 255;
			whiteIntensities.at<Vec3b>(i,j).val[1] = 255;
			whiteIntensities.at<Vec3b>(i,j).val[2] = 255;
			blueIntensities.at<Vec3b>(i,j).val[0] = 128;
			blueIntensities.at<Vec3b>(i,j).val[1] = 128;
			blueIntensities.at<Vec3b>(i,j).val[2] = 0;
			gi.at<Vec3b>(i,j).val[0] =  abs(whiteIntensities.at<Vec3b>(i,j).val[0] - Icolor.at<Vec3b>(i,j).val[0]);
			gi.at<Vec3b>(i,j).val[1] = abs(whiteIntensities.at<Vec3b>(i,j).val[1] - Icolor.at<Vec3b>(i,j).val[1]);
			gi.at<Vec3b>(i,j).val[2] = abs(whiteIntensities.at<Vec3b>(i,j).val[2] - Icolor.at<Vec3b>(i,j).val[2]);
			ge.at<Vec3b>(i,j).val[0] =  abs(blueIntensities.at<Vec3b>(i,j).val[0] - Icolor.at<Vec3b>(i,j).val[0]);
			ge.at<Vec3b>(i,j).val[1] = abs(blueIntensities.at<Vec3b>(i,j).val[1] - Icolor.at<Vec3b>(i,j).val[1]);
			ge.at<Vec3b>(i,j).val[2] = abs(blueIntensities.at<Vec3b>(i,j).val[2] - Icolor.at<Vec3b>(i,j).val[2]);
		}
	}

	Mat giGray, geGray;
	cvtColor(gi, giGray, COLOR_BGR2GRAY);
	cvtColor(ge, geGray, COLOR_BGR2GRAY);

	//Constructing the graph
	Graph<float,float,float> graph(Icolor.rows * Icolor.cols, (Icolor.rows - 1)*Icolor.cols + (Icolor.cols - 1)*Icolor.rows); 
	graph.add_node(Icolor.rows * Icolor.cols);

	for(int i=0;i<Icolor.rows;i++){
		for(int j=0;j<Icolor.cols;j++){
			graph.add_tweights(j + i*Icolor.cols, geGray.at<uchar>(i, j), giGray.at<uchar>(i, j));
		}
	}

	map<pp, bool> visit;

	for(int i=0;i<Icolor.rows;i++){
		for(int j=0;j<Icolor.cols;j++){
			if(i+1 >= 0 && i+1 <Icolor.rows && j+1 >= 0 && j+1 < Icolor.cols && visit.count({{i,j},{i+1,j+1}}) == 0 && visit.count({{i+1,j+1},{i,j}}) == 0){
				graph.add_edge(j + i*Icolor.cols, j+1 + (i+1)*Icolor.cols, (g(G, i, j) + g(G, i+1, j+1))/2, (g(G, i, j) + g(G, i+1, j+1))/2);
				//graph.add_edge(j + i*Icolor.cols, j+1 + (i+1)*Icolor.cols, g(G, (2*i+1)/2, (2*j+1)/2), g(G, (2*i+1)/2, (2*j+1)/2));
				visit[{{i,j},{i+1,j+1}}] = true;
				visit[{{i+1,j+1},{i,j}}] = true;
			}
			if(i-1 >= 0 && i-1 <Icolor.rows && j-1 >= 0 && j-1 < Icolor.cols && visit.count({{i,j},{i-1,j-1}}) == 0 && visit.count({{i-1,j-1},{i,j}}) == 0){
				graph.add_edge(j + i*Icolor.cols, j-1 + (i-1)*Icolor.cols, (g(G, i, j) + g(G, i-1, j-1))/2, (g(G, i, j) + g(G, i-1, j-1))/2);
				//graph.add_edge(j + i*Icolor.cols, j-1 + (i-1)*Icolor.cols, g(G, (2*i-1)/2, (2*j-1)/2), g(G, (2*i-1)/2, (2*j-1)/2));
				visit[{{i,j},{i-1,j-1}}] = true;
				visit[{{i-1,j-1},{i,j}}] = true;
			}
			if(i+1 >= 0 && i+1 <Icolor.rows && j-1 >= 0 && j-1 < Icolor.cols && visit.count({{i,j},{i+1,j-1}}) == 0 && visit.count({{i+1,j-1},{i,j}}) == 0){
				graph.add_edge(j + i*Icolor.cols, j-1 + (i+1)*Icolor.cols, (g(G, i, j) + g(G, i+1, j-1))/2, (g(G, i, j) + g(G, i+1, j-1))/2);
				//graph.add_edge(j + i*Icolor.cols, j-1 + (i+1)*Icolor.cols, g(G, (2*i+1)/2, (2*j-1)/2), g(G, (2*i+1)/2, (2*j-1)/2));
				visit[{{i,j},{i+1,j-1}}] = true;
				visit[{{i+1,j-1},{i,j}}] = true;
			}
			if(i-1 >= 0 && i-1 <Icolor.rows && j+1 >= 0 && j+1 < Icolor.cols && visit.count({{i,j},{i-1,j+1}}) == 0 && visit.count({{i-1,j+1},{i,j}}) == 0){
				graph.add_edge(j + i*Icolor.cols, j+1 + (i-1)*Icolor.cols, (g(G, i, j) + g(G, i-1, j+1))/2, (g(G, i, j) + g(G, i-1, j+1))/2);
				//graph.add_edge(j + i*Icolor.cols, j+1 + (i-1)*Icolor.cols, g(G, (2*i-1)/2, (2*j+1)/2), g(G, (2*i-1)/2, (2*j+1)/2));
				visit[{{i,j},{i-1,j+1}}] = true;
				visit[{{i-1,j+1},{i,j}}] = true;
			}
			
		}
	}

	float flow = graph.maxflow();

	Image<Vec3b> final = Image<Vec3b>(imread("../fishes.jpg"));

	for(int pos=0;pos < Icolor.rows * Icolor.cols;pos++){
		int i = pos/Icolor.cols;
		int j = pos % Icolor.cols;

		if(graph.what_segment(pos) == Graph<float,float,float>::SOURCE){
			final.at<Vec3b>(i,j).val[0] = 255;
			final.at<Vec3b>(i,j).val[1] = 255;
			final.at<Vec3b>(i,j).val[2] = 255;
		}
		else{
			final.at<Vec3b>(i,j).val[0] = 0;
			final.at<Vec3b>(i,j).val[1] = 0;
			final.at<Vec3b>(i,j).val[2] = 0;
		}
	}

	imshow("Final result", final); 

	imshow("Original image",Icolor);

	waitKey(0);
	return 0;
}
