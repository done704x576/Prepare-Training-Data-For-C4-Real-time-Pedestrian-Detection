// C4_SVM_Train_Data.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>
#include<vector>
#include<string>
#include <fstream>
#include <bitset>

using namespace cv;
using namespace std;

void calcFeatures(const Mat &imgSrc, vector<float> &features);
void ComputeSobel(const Mat &gray_image, Mat &sobel_image);
void ComputeCT(const Mat &sobel_image, Mat &CT_image);
void generate_sample_list(int posNum, int negNum);
void generateTrainingData(int nClass, int nDims, int posNum, int negNum);

int _tmain(int argc, _TCHAR* argv[])
{
	int PosSampleNum = 5103;					//正样本个数
	int NegSampleNum = 3102;					//负样本个数
	int nSamples = PosSampleNum + NegSampleNum;	//样本总数
	int nDims = 6144;							//特征维数
	int nClass = 2;								//总类别数

	generateTrainingData(nClass, nDims, PosSampleNum, NegSampleNum);

	waitKey(0);
	return 0;
}

void calcFeatures(const Mat &imgSrc, vector<float> &features)
{
	if (imgSrc.empty())
	{
		cout << "Invalid Input!" << endl;
		return ;
	}

	Mat gray_image(imgSrc.size(), CV_8UC1);
	cvtColor(imgSrc, gray_image, CV_BGR2GRAY);

	Mat sobel_image(gray_image.size(), CV_32FC1);
	ComputeSobel(gray_image, sobel_image);
	//imshow("Sobel-Image", sobel_image);

	Mat CT_feature_image(gray_image.size(), CV_32FC1);
	ComputeCT(sobel_image, CT_feature_image);
	//imshow("CT_feature",CT_feature_image);
	
	//检测窗口的大小为36*108，然后将该检测窗口划分为4*9个block，每个block的大小是9*12
	//每相邻的4个block作为一个super-block，用该super-block来提取CENTRIST(Ct_feature)特征，
	//横向移动步长为9，纵向移动步长为12，每个super-block横向可以移动3下，纵向可以移动8下，
	//一个检测窗口一共可以产生（9-1）*（4-1）= 8*3 = 24个super-block，
	//计算每个super-block的直方图，统计[0-255]共256个特征值每个值出现的次数，最终将生成256*24=6144维的特征。
	int width = 36;
	int height = 108;
	int stepsize = 2;
	int baseflength = 256;	//[0-255]
	int xdiv = 9;
	int ydiv = 12;
	int EXT = 1;

	MatND hist;
	int hist_size[1];
	float hranges[2];
	const float* ranges[1];
	int channels[1];

	hist_size[0] = 256;
	hranges[0] = 0.0;
	hranges[1] = 255.0;
	ranges[0] = hranges;
	channels[0] = 0;

	for (int i = 0; i < height - ydiv; i += ydiv)
	{
		for (int j = 0; j < width - xdiv; j += xdiv)
		{
			Rect super_block_rect(j, i, 2 * xdiv, 2 * ydiv);
			Mat super_block_image = CT_feature_image(super_block_rect);
			calcHist(&super_block_image, 1, channels, Mat(), hist, 1, hist_size, ranges, true, false);
			for (int k = 0; k < 256; k++)
			{
				features.push_back(hist.at<float>(k));
			}
		}
	}
}

void ComputeSobel(const Mat &gray_image, Mat &sobel_image)
{
	for (int i = 1; i < gray_image.rows - 1; i++)
	{
		for (int j = 1; j < gray_image.cols - 1; j++)
		{
			int Gx = (int)gray_image.at<uchar>(i - 1, j - 1) * (-1)+
				(int)gray_image.at<uchar>(i - 1, j) * (-2)+
				(int)gray_image.at<uchar>(i - 1, j + 1) * (-1)+
				(int)gray_image.at<uchar>(i + 1, j - 1)+
				(int)gray_image.at<uchar>(i + 1, j) * 2+
				(int)gray_image.at<uchar>(i + 1, j + 1);

			int Gy = (int)gray_image.at<uchar>(i - 1, j - 1) * (-1)+
				(int)gray_image.at<uchar>(i, j - 1) * (-2)+
				(int)gray_image.at<uchar>(i + 1, j - 1) * (-1)+
				(int)gray_image.at<uchar>(i - 1, j + 1)+
				(int)gray_image.at<uchar>(i, j + 1) * 2+
				(int)gray_image.at<uchar>(i + 1, j + 1);

			float G = (float)(Gx * Gx + Gy * Gy);

			sobel_image.at<float>(i, j) = G;
		}
	}
}

void ComputeCT(const Mat &sobel_image, Mat &CT_image)
{
	for (int i = 2; i < sobel_image.rows - 2; i++)
	{
		for (int j = 2; j < sobel_image.cols - 2; j++)
		{
			int index = 0;
			//if与多个else if，只会执行其中一个条件，这里被自己挖的坑耽误了好几天，现在改写为多个if语句
			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i - 1, j - 1))
			{
				index += 0x80;	//128
			}

			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i - 1, j))
			{
				index += 0x40;	//64
			}

			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i - 1, j + 1))
			{
				index += 0x20;	//32
			}

			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i, j - 1))
			{
				index += 0x10;	//16
			}

			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i, j + 1))
			{
				index += 0x08;	//8
			}

			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i + 1, j - 1))
			{
				index += 0x04;	//4
			}

			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i + 1, j))
			{
				index += 0x02;	//2
			}

			if (sobel_image.at<float>(i, j) <= sobel_image.at<float>(i + 1, j + 1))
			{
				index += 0x01;	//1
			}

			CT_image.at<float>(i, j) = (float)index;
		}
	}
}

void generate_sample_list(int posNum, int negNum)
{
	char imageName[100];
	FILE* pos_fp;
	pos_fp = fopen("PositiveSamplesList.txt","wb+");
	for (int i = 1; i <= posNum; i++)
	{
		sprintf(imageName,"%d.jpg",i);
		fprintf(pos_fp,"%s\r\n",imageName);
	}
	fclose(pos_fp);

	FILE* neg_fp;
	neg_fp = fopen("NegativeSamplesList.txt","wb+");
	for (int i = 1; i <= negNum; i++)
	{
		sprintf(imageName,"%d.jpg",i);
		fprintf(neg_fp,"%s\r\n",imageName);
	}
	fclose(neg_fp);
}

void generateTrainingData(int nClass, int nDims, int posNum, int negNum)
{
	int number = 0;
	int nCount = 0;
	Mat input_image;
	vector<float> features;
	vector<float> labels;

	generate_sample_list(posNum, negNum);//生成正负样本文件名列表
	string ImgName;//图片名(绝对路径)
	ifstream finPos("PositiveSamplesList.txt");//正样本图片的文件名列表
	ifstream finNeg("NegativeSamplesList.txt");//负样本图片的文件名列表

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人

	for(int i = 0; i < posNum && getline(finPos,ImgName); i++)
	{
		ImgName = "..\\pos-2\\" + ImgName;//加上正样本的路径名
		input_image = imread(ImgName);//读取图片
		calcFeatures(input_image, features);
	}
	cout << "Finished processing positive samlpes !" << endl;

	for (int j = 0; j < negNum && getline(finNeg,ImgName); j++)
	{
		ImgName = "..\\neg-2\\" + ImgName;//加上正样本的路径名
		input_image = imread(ImgName);//读取图片
		calcFeatures(input_image, features);
	}
	cout << "Finished processing negative samlpes !" << endl;

	//write the feature data into a txt file, the format must refer to libliner's reference 
	FILE * fp;
	fp = fopen("samples.txt","wb+");;//创建一个txt文件，用于写入数据的，每次写入数据追加到文件尾

	for (int m = 0; m < (posNum + negNum); m++)
	{
		if (m < posNum)
		{
			int lable = 1;		//	positive sample lable 1
			fprintf(fp,"%d ",lable);
		}
		else
		{
			int lable = -1;		//	negative sample lable -1
			fprintf(fp,"%d ",lable); 
		}

		for(int n = 0; n < nDims; n++)
		{
			fprintf(fp,"%d:%f ",(n+1),features.at(m * nDims + n));
		}
		fprintf(fp,"\r\n");
	}

	cout << "Generate Training Data Complete!" << endl << endl;
}



