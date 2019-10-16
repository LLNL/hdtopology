#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

/*
Load raw binary file with meta information in text form, assume the binary file consists of double (64bit)

format:


binary_file_name: xxx.bin
num_samples: 10000000
num_scalars: 3
num_inputs: 5
scalars:
	ae_loss
	fw_latent
	fw_output
inputs:
	shape_model_initial_modes:(4,3)
	betti_prl15_trans_u
	betti_prl15_trans_v
	shape_model_initial_modes:(2,1)
	shape_model_initial_modes:(1,0)

*/

// trim helper
std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    return ltrim(rtrim(str, chars), chars);
}



// load from binary file to fill the buffer
/*
binPath              input binary file name
labelPath            description of the binary
scalarName           the function name
N                    sample size
D                    function domain dim

//input are assume to be 64bit double

*/

bool loadBin(std::string labelPath, std::string scalarName, std::vector<float> &buffer, int &N, int &D, std::vector<std::string> &attributes, int subsampleCount = -1){
  //compute the index of the scalar
  std::ifstream input(labelPath);
  std::vector<std::string> scalarLabel;
  std::string line;
  std::string binPath;
  int scalarIndex, inputStartIndex;
  int D_scalar;

  int count = 0;
  int countScalar = 0;
  bool scalarStart = false;
  bool inputStart = false;

  if (input.is_open()){
    fprintf(stderr, "attributes:\n");
    while(getline(input, line)){
      std::istringstream iss(line);
      std::string tag;

      fprintf(stderr, " %s\n", trim(line).c_str());

      if(scalarStart){
          scalarLabel.push_back(trim(line));
          if(trim(line) == scalarName)
            scalarIndex = count;
          count++;
      }

      if(inputStart){
          attributes.push_back(trim(line));
      }

      while(iss >> tag){
        if(tag == std::string("num_samples:"))
          iss >> N;
        else if(tag == std::string("num_inputs:"))
          iss >> D;
        else if(tag == std::string("num_scalars:"))
          iss >> D_scalar;
        else if(tag == std::string("binary_file_name:"))
          iss >> binPath;
        else if(tag == std::string("scalars:")){
          scalarStart = true;
          inputStart = false;
        }
        else if(tag == std::string("inputs:")){
          inputStart = true;
          scalarStart = false;
        }
      }
    }

    //get index where the input start in scalar
    for(size_t i=0; i<scalarLabel.size(); i++)
      if(scalarLabel[i] == attributes[0])
        inputStartIndex = i;

    fprintf(stderr, "num_samples:%d num_inputs:%d num_scalars:%d \n", N, D, D_scalar);
    fprintf(stderr, "scalarIndex:%d inputStartIndex:%d scalarName:%s\n ", scalarIndex, inputStartIndex, scalarName.c_str());
    attributes.push_back(scalarLabel[scalarIndex]);
  }
  else
    return false;

  //load the data
  std::ifstream binInput(binPath, std::ios::binary| std::ios::in| std::ios::ate);
  if (binInput.is_open())
  {
      buffer.clear();
      size_t size = binInput.tellg();
      std::cout << "size = " << size << std::endl;
      char* memblock = new char[size];
      binInput.seekg (0, std::ios::beg);
      binInput.read (memblock, size);
      binInput.close();

      if(subsampleCount<N && subsampleCount!=-1)
        N = subsampleCount;

      double *data = (double*)memblock;
      for(size_t i=0; i<N; i++){
        for(size_t j=0; j<D; j++){
          buffer.push_back(data[ i*(D_scalar) + inputStartIndex + j ]);
        }
        buffer.push_back(data[ i*(D_scalar) + scalarIndex ]);
     }
      std::cout<<"buffer size: "<<buffer.size()<<std::endl;
      return true;
  }

  return false;
}
