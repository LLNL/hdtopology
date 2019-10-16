#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

/*
Load raw binary file with meta information in text form, assume the binary file consists of double (64bit), input is before the outputs

format:

binary_file_name: xxx.bin
num_samples: 10000000
num_scalars: 3
num_inputs: 5
inputs:
	shape_model_initial_modes:(4,3)
	betti_prl15_trans_u
	betti_prl15_trans_v
	shape_model_initial_modes:(2,1)
	shape_model_initial_modes:(1,0)
outputs:
	ae_loss
	fw_latent
	fw_output


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
  std::vector<std::string> outputLabel;
  std::string line;
  std::string binPath;
  int outputIndex = -1;
  int inputStartIndex = 0;
  int D_output = -1;

  int count = 0;
  int countScalar = 0;
  bool outputStart = false;
  bool inputStart = false;

  if (input.is_open()){
    fprintf(stderr, "attributes:\n");
    while(getline(input, line)){
      std::istringstream iss(line);
      std::string tag;

      fprintf(stderr, " %s\n", trim(line).c_str());

      if(outputStart){
          outputLabel.push_back(trim(line));
          if(trim(line) == scalarName)
            outputIndex = count;
          count++;
      }

      if(inputStart && trim(line)!=std::string("outputs:")){
          attributes.push_back(trim(line));
      }

      while(iss >> tag){
        if(tag == std::string("num_samples:"))
          iss >> N;
        else if(tag == std::string("num_inputs:"))
          iss >> D;
        else if(tag == std::string("num_outputs:"))
          iss >> D_output;
        else if(tag == std::string("binary_file_name:"))
          iss >> binPath;
        else if(tag == std::string("inputs:")){
          inputStart = true;
          outputStart = false;
        }
        else if(tag == std::string("outputs:")){
          outputStart = true;
          inputStart = false;
        }
      }
    }

    fprintf(stderr, "num_samples:%d num_inputs:%d num_outputs:%d \n", N, D, D_output);
    fprintf(stderr, "outputIndex:%d inputStartIndex:%d scalarName:%s\n ", outputIndex, inputStartIndex, scalarName.c_str());
    attributes.push_back(outputLabel[outputIndex]);
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

      double *data = (double*)memblock;
      for(size_t i=0; i<N; i++){
        for(size_t j=0; j<D; j++){
          buffer.push_back(data[ i*(D+D_output) + inputStartIndex + j ]);
        }
        buffer.push_back(data[ i*(D+D_output) + D + outputIndex ]);
     }
      std::cout<<"buffer size: "<<buffer.size()<<std::endl;
      return true;
  }

  return false;
}
