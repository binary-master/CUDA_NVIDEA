// ConsoleApplication2.cpp : Defines the entry point for the console application.
//



#include<iostream>
#include<math.h>
#include<conio.h>
#include<stdlib.h>
#include<vector>
#include<algorithm>
#include<map>
#include<iterator>
#include <fstream>
#include <streambuf>
#include<string>
#include <dirent.h>
#include <boost/algorithm/string.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <random>


int D = 10000;
int N = 3;

std::vector<int> genRandomHV()
{
	std::vector<int> randomIndex(D);
	std::vector<int> randomHV(D);
	std::mt19937 r{ std::random_device{}() };
	if ((D % 2) == 1)
	{
		std::cout << "Dimension is odd";
	}
	else
	{
		for (int i = 0; i < D; i++)
			randomIndex[i] = i;
		std::shuffle(randomIndex.begin(), randomIndex.end(), r);
		for (int i = 0;i < D / 2;i++)
			randomHV[randomIndex[i]] = 1;
		for (int i = D / 2;i < D;i++)
			randomHV[randomIndex[i]] = -1;
	}
	
	return randomHV;
}

std::map<char, std::vector<int>> createItemMemory(std::map<char, std::vector<int>> iM)
{
	
		
	iM['a'] = genRandomHV();
	iM['b'] = genRandomHV();
	iM['c'] = genRandomHV();
	iM['d'] = genRandomHV();
	iM['e'] = genRandomHV();
	iM['f'] = genRandomHV();
	iM['g'] = genRandomHV();
	iM['h'] = genRandomHV();
	iM['i'] = genRandomHV();
	iM['j'] = genRandomHV();
	iM['k'] = genRandomHV();
	iM['l'] = genRandomHV();
	iM['m'] = genRandomHV();
	iM['n'] = genRandomHV();
	iM['o'] = genRandomHV();
	iM['p'] = genRandomHV();
	iM['q'] = genRandomHV();
	iM['r'] = genRandomHV();
	iM['s'] = genRandomHV();
	iM['t'] = genRandomHV();
	iM['u'] = genRandomHV();
	iM['v'] = genRandomHV();
	iM['w'] = genRandomHV();
	iM['x'] = genRandomHV();
	iM['y'] = genRandomHV();
	iM['z'] = genRandomHV();
	iM[char(32)] = genRandomHV();
	
	return iM;
}

std::vector<int> lookUpitemMemory(std::map<char, std::vector<int>> iM, char key)
{
	std::vector<int> randomHV(D);
	randomHV = iM[key];
	return randomHV;
}

double cosine_similarity(std::vector<int> A, std::vector<int> B)
{
	
	double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
	for (int i = 0; i < D; ++i) {
		dot += A[i] * B[i];
		denom_a += A[i] * A[i];
		denom_b += B[i] * B[i];
	}
	return (dot / (sqrt(denom_a) * sqrt(denom_b)));
}

std::vector<int> binarizeHV(std::vector<int> langHV)
{
	int threshold = 0;
	/*for (size_t i = 0; i < langHV.size(); i++) {
		std::cout << langHV[i] << std::endl;
	}*/
	for (int i = 0; i < D; i++)
	{
		if (langHV[i] > threshold)
		{
			langHV[i] = 1;
		}
		else
		{
			langHV[i] = -1;
		}
	}
	
	/*for (size_t i = 0; i < langHV.size(); i++) {
		std::cout << langHV[i] << std::endl;
		}*/

	return langHV;
}

std::vector<int> computeSumHV(std::map<char, std::vector<int>> iM, size_t bufferSize, std::string  buffer)
{
	//std::vector<int> h_block0(D, 1);
	thrust::device_vector<int> st_block2(D, 1);
	thrust::device_vector<int> st_block3(D, 1);
	thrust::device_vector<int> st_block4(D, 1);
	thrust::device_vector<int> st_block5(D, 1);
	thrust::device_vector<int> st_block6(D, 1);
	thrust::device_vector<int> st_block7(D, 1);
	thrust::device_vector<int> block0(D, 1);
	thrust::device_vector<int> block1(D, 1);
	thrust::device_vector<int> block2(D, 1);
	thrust::device_vector<int> block3(D, 1);
	thrust::device_vector<int> block4(D, 1);
	thrust::device_vector<int> block5(D, 1);
	thrust::device_vector<int> block6(D, 1);
	thrust::device_vector<int> block7(D, 1);
	thrust::device_vector<int> nGrams(D, 1);
	thrust::device_vector <int > d_sumHV(D, 0);
	std::vector<int> sumHV(D, 0);
	if (N == 3)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];
			/*for (size_t i = 0; i < block0.size(); i++) {
			block0[i] = block1[i];
			}*/
			std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			//rotate(h_block1.begin(), h_block1.end() - 1, h_block0.end());
			/*for (size_t i = 0; i < block1.size(); i++) {
			block1[i] = block2[i];
			}*/
			//block1 = block2; 
			//rotate(block1.begin(), block1.end() - 1, block1.end());
			st_block2 = lookUpitemMemory(iM, key);

			thrust::copy(st_block2.begin(), st_block2.end(), block2.begin());

			if (j >= 2)
			{
				thrust::copy(block2.begin(), block2.end(), nGrams.begin());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] = block2[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block1[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	std::cout << nGrams[i] << std::endl;
				//}
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block0[i];
				//}
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	sumHV[i] += nGrams[i];
				//}

			}

		}

	}

	else if (N == 4)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];
			/*for (size_t i = 0; i < block0.size(); i++) {
			block0[i] = block1[i];
			}*/
			//std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			//rotate(h_block1.begin(), h_block1.end() - 1, h_block0.end());
			/*for (size_t i = 0; i < block1.size(); i++) {
			block1[i] = block2[i];
			}*/
			//block1 = block2; 
			//rotate(block1.begin(), block1.end() - 1, block1.end());
			st_block3 = lookUpitemMemory(iM, key);

			thrust::copy(st_block3.begin(), st_block3.end(), block3.begin());

			if (j >= 3)
			{
				thrust::copy(block3.begin(), block3.end(), nGrams.begin());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] = block2[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block1[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	std::cout << nGrams[i] << std::endl;
				//}
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block0[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());

				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	sumHV[i] += nGrams[i];
				//}

			}

		}

	}

	else if (N == 5)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];
			/*for (size_t i = 0; i < block0.size(); i++) {
			block0[i] = block1[i];
			}*/
			std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			//rotate(h_block1.begin(), h_block1.end() - 1, h_block0.end());
			/*for (size_t i = 0; i < block1.size(); i++) {
			block1[i] = block2[i];
			}*/
			//block1 = block2; 
			//rotate(block1.begin(), block1.end() - 1, block1.end());
			st_block4 = lookUpitemMemory(iM, key);

			thrust::copy(st_block4.begin(), st_block4.end(), block4.begin());

			if (j >= 4)
			{
				thrust::copy(block4.begin(), block4.end(), nGrams.begin());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] = block2[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block1[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	std::cout << nGrams[i] << std::endl;
				//}
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block0[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());


				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	sumHV[i] += nGrams[i];
				//}

			}

		}

	}

	else if (N == 6)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];
			/*for (size_t i = 0; i < block0.size(); i++) {
			block0[i] = block1[i];
			}*/
			//std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			//rotate(h_block1.begin(), h_block1.end() - 1, h_block0.end());
			/*for (size_t i = 0; i < block1.size(); i++) {
			block1[i] = block2[i];
			}*/
			//block1 = block2; 
			//rotate(block1.begin(), block1.end() - 1, block1.end());
			st_block5 = lookUpitemMemory(iM, key);

			thrust::copy(st_block5.begin(), st_block5.end(), block5.begin());

			if (j >= 5)
			{
				thrust::copy(block5.begin(), block5.end(), nGrams.begin());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] = block2[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block1[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	std::cout << nGrams[i] << std::endl;
				//}
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block0[i];
				//}
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	sumHV[i] += nGrams[i];
				//}

			}

		}

	}

	else if (N == 7)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];
			/*for (size_t i = 0; i < block0.size(); i++) {
			block0[i] = block1[i];
			}*/
			//std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			thrust::copy(block6.begin() + 1, block6.end(), block5.begin());

			//rotate(h_block1.begin(), h_block1.end() - 1, h_block0.end());
			/*for (size_t i = 0; i < block1.size(); i++) {
			block1[i] = block2[i];
			}*/
			//block1 = block2; 
			//rotate(block1.begin(), block1.end() - 1, block1.end());
			st_block6 = lookUpitemMemory(iM, key);

			thrust::copy(st_block6.begin(), st_block6.end(), block6.begin());

			if (j >= 6)
			{
				thrust::copy(block6.begin(), block6.end(), nGrams.begin());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] = block2[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block5.begin(), thrust::multiplies<int>());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block1[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	std::cout << nGrams[i] << std::endl;
				//}
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block0[i];
				//}
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	sumHV[i] += nGrams[i];
				//}

			}

		}

	}

	else if (N == 8)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];
			/*for (size_t i = 0; i < block0.size(); i++) {
			block0[i] = block1[i];
			}*/
			//std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			thrust::copy(block6.begin() + 1, block6.end(), block5.begin());
			thrust::copy(block7.begin() + 1, block7.end(), block6.begin());

			//rotate(h_block1.begin(), h_block1.end() - 1, h_block0.end());
			/*for (size_t i = 0; i < block1.size(); i++) {
			block1[i] = block2[i];
			}*/
			//block1 = block2; 
			//rotate(block1.begin(), block1.end() - 1, block1.end());
			st_block7 = lookUpitemMemory(iM, key);

			thrust::copy(st_block7.begin(), st_block7.end(), block7.begin());

			if (j >= 7)
			{
				thrust::copy(block7.begin(), block7.end(), nGrams.begin());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] = block2[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block6.begin(), thrust::multiplies<int>());

				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block1[i];
				//}
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block5.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	std::cout << nGrams[i] << std::endl;
				//}
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	nGrams[i] *= block0[i];
				//}
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
				//for (size_t i = 0; i < nGrams.size(); i++) {
				//	sumHV[i] += nGrams[i];
				//}

			}

		}

	}
	thrust::copy(d_sumHV.begin(), d_sumHV.end(), sumHV.begin());

	return sumHV;

}

std::map<std::string, std::vector<int>> buildLanguage(std::map<char, std::vector<int>> iM)
{
	std::map< std::string, std::vector<int>> langAM;
	std::vector<int> langHV(D);
	//size_t size = 0;
	//char *ch = NULL;
	//FILE *file = NULL;
	/*errno_t err;*/

	std::string langLabels[8];
	langLabels[0] = "acq";
	langLabels[1] = "cru";
	langLabels[2] = "ear";
	langLabels[3] = "gra";
	langLabels[4] = "int";
	langLabels[5] = "mon";
	langLabels[6] = "shi";
	langLabels[7] = "tra";
	//std::string langText;
	//langText = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\acq.txt";
	/*langText[1] = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\cru.txt";
	langText[2] = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\gra.txt";
	langText[3] = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\int.txt";
	langText[4] = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\mon.txt";
	langText[5] = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\ear.txt";
	langText[6] = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\shi.txt";
	langText[7] = "C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\tra.txt";
	*/
	for (int i = 0; i < 8; i++)
	{

		/*int TempNumOne = langLabels[i].size();
		char Filename[100];
		for (int a = 0;a <= TempNumOne;a++)
		{
			Filename[a] = langText[a];
		}
		*/

		switch (i)
		{
		case 0: {std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\acq.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV (langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}
		case 1: {
			std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\cru.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV(langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}
		case 2: {
			std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\ear.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV(langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}
		case 3: {
			std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\gra.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV(langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}
		case 4: {
			std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\int.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV(langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}
		case 5: {
			std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\mon.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV(langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}
		case 6: {
			std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\shi.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV(langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}
		case 7: {
			std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\tra.txt");
			std::string str;

			/*t.seekg(0, std::ios::end);
			str.reserve(t.tellg());
			t.seekg(0, std::ios::beg);*/

			/*std::ifstream t("file.txt");*/
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			/*fseek(file, 0, SEEK_END);
			size = ftell(file);
			rewind(file);
			ch = (char *)malloc((size + 1) * sizeof(ch));
			fread(ch, size, 1, file);*/
			std::cout << "Training File:" << langLabels[i] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[i])] = binarizeHV(langHV);
			//langAM[(langLabels[i])] = binarizeHV(langAM[(langLabels[i])]);
			break;
		}

				/*}
					std::ifstream t("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\acq.txt");
					std::string str;

					t.seekg(0, std::ios::end);
					str.reserve(t.tellg());
					t.seekg(0, std::ios::beg);

					std::ifstream t("file.txt");
					t.seekg(0, std::ios::end);
					size_t size = t.tellg();
					std::string buffer(size, ' ');
					t.seekg(0);
					t.read(&buffer[0], size);

					/*fseek(file, 0, SEEK_END);
					size = ftell(file);
					rewind(file);
					ch = (char *)malloc((size + 1) * sizeof(ch));
					fread(ch, size, 1, file);
					std::cout << size;

					langHV = computeSumHV(iM, size, buffer);
					langAM[(langLabels[i])] = langHV;

					/*printf("%s\n", buffer.c_str());*/

		}
	}
	return langAM;

	/*
	for(int i=0;i<size;i++)
	{
	printf("%c",ch[i]);
	}
	*/
	
	
}


double test(std::map<char, std::vector<int>> iM, std::map<std::string, std::vector<int>> langAM)
{
	double total = 0.0;
	double correct = 0.0;
	double accuracy = 0;
	double maxAngle, angle = 0;
	std::string predictLang;
	std::vector<int> textHV;

	std::string langLabels[8];
	langLabels[0] = "acq";
	langLabels[1] = "cru";
	langLabels[2] = "gra";
	langLabels[3] = "int";
	langLabels[4] = "mon";
	langLabels[5] = "ear";
	langLabels[6] = "shi";
	langLabels[7] = "tra";

	DIR *pdir = NULL; // remember, it's good practice to initialise a pointer to NULL!

	    pdir = opendir ("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication2\\ConsoleApplication2\\Debug\\testing_texts"); // "." will refer to the current directory

	    struct dirent *pent = NULL;

	 

	    // I used the current directory, since this is one which will apply to anyone reading

	    // this tutorial~ If I said "C:\\" and you're on Linux, it may get a little confusing!

	    if (pdir == NULL) // if pdir wasn't initialised correctly

	    { // print an error message and exit the program

	        std::cout << "\nERROR! pdir could not be initialised correctly";

	        exit (3);

	    } // end if

	 

	    while (pent = readdir (pdir)) // while there is still something in the directory to list

	    {

	        if (pent == NULL) // if pent has not been initialised correctly

	        { // print an error message, and exit the program

	            std::cout << "\nERROR! pent could not be initialised correctly";

	            exit (3);

	        }
			if (strcmp(pent->d_name, ".") != 0 && strcmp(pent->d_name, "..") != 0)
	        // otherwise, it was initialised correctly. Let's print it on the console:
			//if (pent->d_name == ".")
			//{
				//continue;
			//}
			//else
			{
				std::cout << pent->d_name << std::endl;
				std::string name = pent->d_name;
				std::string actualLabel = name.substr(0, 3);
				std::vector<std::string> list{ "C:", "Users", "Mohammed Aashyk", "Documents", "Visual Studio 2015", "Projects", "ConsoleApplication2", "ConsoleApplication2", "Debug", "testing_texts", name };
				//std::cout << name;
				std::string joined = boost::algorithm::join(list, "\\");
				//std::cout << joined;
				std::ifstream t(joined);
				std::string str;

				/*t.seekg(0, std::ios::end);
				str.reserve(t.tellg());
				t.seekg(0, std::ios::beg);*/

				/*std::ifstream t("file.txt");*/
				t.seekg(0, std::ios::end);
				size_t size = t.tellg();
				std::string buffer(size, ' ');
				t.seekg(0);
				t.read(&buffer[0], size);

				/*fseek(file, 0, SEEK_END);
				size = ftell(file);
				rewind(file);
				ch = (char *)malloc((size + 1) * sizeof(ch));
				fread(ch, size, 1, file);*/
				

				std::cout << "Loading test file:" << pent->d_name << std::endl;
				textHV = computeSumHV(iM, size, buffer);
				textHV = binarizeHV(textHV);
				maxAngle = -1;
				for (int i = 0; i < 8; i++)
				{
					angle = cosine_similarity(langAM[langLabels[i]], textHV);
					if (angle > maxAngle)
					{
						maxAngle = angle;
						predictLang = langLabels[i];
					}
					

				}
				if (predictLang == actualLabel)
				{
					correct = correct + 1.0;
				}
				else
				{
					std::cout << predictLang << "  -->  " << actualLabel <<  std::endl;
				}
			}

			total = total + 1.0;
	    }

		//std::cout << pent->d_name;
		
		

	    // finally, let's close the directory

	    closedir (pdir);

	 

	    //std::cin.get (); // pause for input

	   // return EXIT_SUCCESS; // everything went OK
		accuracy = correct / total * 100;

		return accuracy;
}


void printPair(const std::pair<char, std::vector<int> > &p)
{
	std::cout << "Key: " << p.first << std::endl;
	copy(p.second.begin(), p.second.end(), std::ostream_iterator<int>(std::cout, "\t"));
}
int main()
{
	
	std::vector<int> rand;
	std::map<char, std::vector<int>> iM;
	std::map<std::string, std::vector<int>> langAM;
	double correct;
	iM = createItemMemory(iM);

	//for_each(iM.begin(), iM.end(), printPair);
	//std::cout << "rand contains:";
	//for (std::vector<int>::iterator it = rand.begin(); it != rand.end(); ++it)
	//	std::cout << ' ' << *it;

	langAM = buildLanguage(iM);
	/*std::map<std::string, std::vector<int>>::iterator pos;
	for (pos = langAM.begin(); pos != langAM.end(); ++pos) {
	std::cout << "key: \"" << pos->first << "\" " << std::endl << "values: \"" ;
	typedef std::vector<int>::const_iterator ListIterator;
	for (ListIterator list_iter = pos->second.begin(); list_iter != pos->second.end(); list_iter++)
	std::cout << " " << *list_iter << std::endl;
	}
	/* for (std::map<std::string, std::vector<long int>> ::const_iterator it = langAM.begin();//
	it != langAM.end(); ++it)
	{
	std::cout << it->first << " " << it->second.first << " " << it->second.second << "\n";
	}*/
	
	correct = test(iM, langAM);

	/*std::map<std::string, std::vector<int>>::iterator pos;
	for (pos = langAM.begin(); pos != langAM.end(); ++pos) {
		std::cout << "key: \"" << pos->first << "\" " << std::endl << "values: \"" ;
		typedef std::vector<int>::const_iterator ListIterator;
		for (ListIterator list_iter = pos->second.begin(); list_iter != pos->second.end(); list_iter++)
			std::cout << " " << *list_iter << std::endl;
	}
	/* for (std::map<std::string, std::vector<long int>> ::const_iterator it = langAM.begin();
		it != langAM.end(); ++it)
	{
		std::cout << it->first << " " << it->second.first << " " << it->second.second << "\n";
	}*/
	
	std::cout << correct << "%" << std::endl << "Run Success!";


}



