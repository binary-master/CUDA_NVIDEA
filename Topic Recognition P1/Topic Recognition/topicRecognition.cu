//Topic Recognition P5 

//include Header and Libraries files required
//note: boost library must be installed and included in the Additional Include Directories of the project Properties for this code to work.
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

//setting the parameters for size of HD vectors and N-Grams
int D = 10000;
int N = 3;

//generate random HyperVectors
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
		std::shuffle(randomIndex.begin(), randomIndex.end(), r);		//shuffles the random vectors positions
		for (int i = 0;i < D / 2;i++)
			randomHV[randomIndex[i]] = 1;
		for (int i = D / 2;i < D;i++)
			randomHV[randomIndex[i]] = -1;
	}

	return randomHV;
}


//create the Item Memory from which the values will be drrived
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

//Retrieve the Character's corresponding Hyper vector
std::vector<int> lookUpitemMemory(std::map<char, std::vector<int>> iM, char key)
{
	std::vector<int> randomHV(D);
	randomHV = iM[key];
	return randomHV;
}

//Finding similarity between Two Hyper vectors
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

//Converting the vectors into values +1 and -1 before comparison(cosine)
std::vector<int> binarizeHV(std::vector<int> langHV)
{
	int threshold = 0;

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

	return langHV;
}

//Main part of the program: Generates the hyper vector for each input text file
//thrust algorithm is used which allocates the memory in the device (i.e, GPU) and performs all operations in the GPU
std::vector<int> computeSumHV(std::map<char, std::vector<int>> iM, size_t bufferSize, std::string  buffer)
{
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

			std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			;
			st_block2 = lookUpitemMemory(iM, key);

			thrust::copy(st_block2.begin(), st_block2.end(), block2.begin());

			if (j >= 2)
			{
				thrust::copy(block2.begin(), block2.end(), nGrams.begin());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());

				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());

			}

		}

	}

	else if (N == 4)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());

			st_block3 = lookUpitemMemory(iM, key);

			thrust::copy(st_block3.begin(), st_block3.end(), block3.begin());

			if (j >= 3)
			{
				thrust::copy(block3.begin(), block3.end(), nGrams.begin());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());

				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
			}

		}

	}

	else if (N == 5)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];

			std::cout << key;

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			;
			st_block4 = lookUpitemMemory(iM, key);

			thrust::copy(st_block4.begin(), st_block4.end(), block4.begin());

			if (j >= 4)
			{
				thrust::copy(block4.begin(), block4.end(), nGrams.begin());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());

				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());

			}

		}

	}

	else if (N == 6)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());

			st_block5 = lookUpitemMemory(iM, key);

			thrust::copy(st_block5.begin(), st_block5.end(), block5.begin());

			if (j >= 5)
			{
				thrust::copy(block5.begin(), block5.end(), nGrams.begin());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());

				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());

			}

		}

	}

	else if (N == 7)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			thrust::copy(block6.begin() + 1, block6.end(), block5.begin());

			st_block6 = lookUpitemMemory(iM, key);

			thrust::copy(st_block6.begin(), st_block6.end(), block6.begin());

			if (j >= 6)
			{
				thrust::copy(block6.begin(), block6.end(), nGrams.begin());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block5.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());

				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());

			}

		}

	}

	else if (N == 8)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int i = j;
			char key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			thrust::copy(block6.begin() + 1, block6.end(), block5.begin());
			thrust::copy(block7.begin() + 1, block7.end(), block6.begin());

			st_block7 = lookUpitemMemory(iM, key);

			thrust::copy(st_block7.begin(), st_block7.end(), block7.begin());

			if (j >= 7)
			{
				thrust::copy(block7.begin(), block7.end(), nGrams.begin());

			}
			thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block6.begin(), thrust::multiplies<int>());

			thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block5.begin(), thrust::multiplies<int>());

			thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());

			thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());

			thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());

			thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());

			thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());

			thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());

		}

	}


thrust::copy(d_sumHV.begin(), d_sumHV.end(), sumHV.begin());

return sumHV;

}

//Builds Associative Memory From the training Files
std::map<std::string, std::vector<int>> buildLanguage(std::map<char, std::vector<int>> iM)
{
	std::map< std::string, std::vector<int>> langAM;
	std::vector<int> langHV(D);

	int count = 0;
	std::string langLabels[64];

	DIR *pdir = NULL;

	pdir = opendir("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\Topic Recognition P2\\Topic Recognition P2\\Training Files P2"); // "." will refer to the current directory

	struct dirent *pent = NULL;


	if (pdir == NULL)

	{
		std::cout << "\nERROR! pdir could not be initialised correctly";

		exit(3);

	} // end if



	while (pent = readdir(pdir))

	{

		if (pent == NULL)

		{

			std::cout << "\nERROR! pent could not be initialised correctly";

			exit(3);

		}
		if (strcmp(pent->d_name, ".") != 0 && strcmp(pent->d_name, "..") != 0)

		{
			std::cout << pent->d_name << std::endl;
			std::string name = pent->d_name;
			langLabels[count] = name.substr(0, 4);

			std::vector<std::string> list{ "C:", "Users", "Mohammed Aashyk", "Documents", "Visual Studio 2015", "Projects", "Topic Recognition P2", "Topic Recognition P2", "Training Files P2", name };

			std::string joined = boost::algorithm::join(list, "\\");

			std::ifstream t(joined);

			std::string str;

			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			std::cout << "Training File:" << langLabels[count] << std::endl;

			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[count])] = binarizeHV(langHV);

			count += 1;



		}

	}
	return langAM;

}


//Recognize the hypervectors by comparing it with Associative Memory
double test(std::map<char, std::vector<int>> iM, std::map<std::string, std::vector<int>> langAM)
{
	double total = 0.0;
	double correct = 0.0;
	double accuracy = 0;
	double maxAngle, angle = 0;
	std::string predictLang;
	std::vector<int> textHV;

	std::string langLabels[64];
	langLabels[0] = "acq";
	langLabels[8] = "cru";
	langLabels[16] = "gra";
	langLabels[24] = "int";
	langLabels[32] = "mon";
	langLabels[40] = "ear";
	langLabels[48] = "shi";
	langLabels[56] = "tra";


	DIR *pdir = NULL;
	pdir = opendir("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\Topic Recognition P2\\Topic Recognition P2\\testing_texts"); // "." will refer to the current directory

	struct dirent *pent = NULL;

	if (pdir == NULL)

	{

		std::cout << "\nERROR! pdir could not be initialised correctly";

		exit(3);

	}



	while (pent = readdir(pdir))

	{

		if (pent == NULL)

		{

			std::cout << "\nERROR! pent could not be initialised correctly";

			exit(3);

		}
		if (strcmp(pent->d_name, ".") != 0 && strcmp(pent->d_name, "..") != 0)

		{
			std::cout << pent->d_name << std::endl;
			std::string name = pent->d_name;
			std::string actualLabel = name.substr(0, 3);
			std::vector<std::string> list{ "C:", "Users", "Mohammed Aashyk", "Documents", "Visual Studio 2015", "Projects", "Topic Recognition P2", "Topic Recognition P2", "testing_texts", name };

			std::string joined = boost::algorithm::join(list, "\\");

			std::ifstream t(joined);
			std::string str;

			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

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
					predictLang = langLabels[i].substr(0, 3);
				}


			}
			if (predictLang == actualLabel)
			{
				correct = correct + 1.0;
			}
			else
			{
				std::cout << predictLang << "  -->  " << actualLabel << std::endl;
			}
		}

		total = total + 1.0;
	}


	closedir(pdir);

	accuracy = correct / total * 100;

	return accuracy;
}


int main()
{

	std::vector<int> rand;
	std::map<char, std::vector<int>> iM;
	std::map<std::string, std::vector<int>> langAM;
	double correct;
	iM = createItemMemory(iM);										//creates Item Memory to initaiate the program
	langAM = buildLanguage(iM);										//Builds the associative memory from the train files
	correct = test(iM, langAM);										//Compares the test documents with the associative memory
	std::cout << correct << "%" << std::endl << "Run Success!";
	//Displays Accuracy
}


