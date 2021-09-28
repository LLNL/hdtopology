#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#ifndef _MSC_VER
  #include <sys/time.h>
  #include <unistd.h>
#else
  #include <io.h>
#endif

#include <iomanip>
#include <cmath>
#include <ostream>

#include <HDData.h>
#include <ngl.h>
using namespace ngl;

#include <ExtremumGraph.h>
#include <Graph.h>

#ifdef ENABLE_STREAMING
#include <ANNSearchIndex.h>
#endif

#include <NGLIterator.h>

#include <Neighborhood.h>
#include <NeighborhoodIterator.h>
#include <DataCollectionHandle.h>
#include <DataBlockHandle.h>


#ifdef ENABLE_FLANN
  #include <FLANNSearchIndex.h>
#endif

#ifdef ENABLE_FAISS
  #include <FAISSSearchIndex.h>
#endif

#include <DataCollectionHandle.h>
#include <DataBlockHandle.h>
#include "binaryFileLoader.h"

#include <ostream>

#ifdef _MSC_VER
#include <Windows.h>

int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
  // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
  // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
  // until 00:00:00 January 1, 1970
  static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

  SYSTEMTIME  system_time;
  FILETIME    file_time;
  uint64_t    time;

  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  time = ((uint64_t)file_time.dwLowDateTime);
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  tp->tv_sec = (long)((time - EPOCH) / 10000000L);
  tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
  return 0;
}

#endif

namespace Color {
    enum Code {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };
}

typedef struct timeval timestamp;
static inline float operator - (const timestamp &t1, const timestamp &t2)
{
    return (float)(t1.tv_sec  - t2.tv_sec) + 1.0e-6f*(t1.tv_usec - t2.tv_usec);
}
static inline timestamp now()
{
    timestamp t;
    gettimeofday(&t, NULL);
    return t;
}

class CommandLine {
	std::vector<std::string> argnames;
  std::map<std::string, std::string> args;
	std::map<std::string, std::string> descriptions;
	std::vector<std::string> requiredArgs;
  public:
    CommandLine() { }
    void addArgument(std::string cmd, std::string defaultValue, std::string description="", bool required = false) {
      argnames.push_back(cmd);
      if(description!="") {
          descriptions[cmd] = description;
      }
      if(required && find(requiredArgs.begin(), requiredArgs.end(), cmd)==requiredArgs.end()) {
        requiredArgs.push_back(cmd);
      }
      setArgument(cmd, defaultValue);
    }
    void setArgument(std::string cmd, std::string value) {
      args[cmd] = value;
    }
    bool processArgs(int argc, char *argv[]) {
      std::map<std::string, int> check;
      for(unsigned int k = 0;k<requiredArgs.size();k++) {
        check[requiredArgs[k]] = 0;
      }
      for(int i=1; i<argc; i+=2) {
        if(i+1<argc) {
          std::string arg = std::string(argv[i]);
          std::string val = std::string(argv[i+1]);
          setArgument(arg, val);
          if(check.find(arg)!=check.end()) {
            check[arg] = check[arg] + 1;
          }
        }
      }
      for(std::map<std::string,int>::const_iterator it = check.begin(); it!=check.end();it++) {
        int n = it->second;
              if(n==0) {
                  return false;
              }
      }
      return true;
    }
    void showUsage() {
      for(unsigned int k = 0; k<argnames.size();k++) {
        fprintf(stderr, "\t%s\t\t%s (%s)\n", argnames[k].c_str(), descriptions[argnames[k]].c_str(), args[argnames[k]].c_str());
      }
    }
    float getArgFloat(std::string arg) {
      std::string val = args[arg];
      return atof(val.c_str());
    }
    int getArgInt(std::string arg) {
          std::string val = args[arg];
      return atoi(val.c_str());
    }
    std::string getArgString(std::string arg) {
      std::string val = args[arg];
      return val;
    }
};

int main(int argc, char **argv)
{
  timestamp t1 = now();
  timestamp t2;
  CommandLine cl;

  // cl.addArgument("-c", "1000000", "Number of points", true);
  cl.addArgument("-a", "-1", "Number of attributes", false);
  // cl.addArgument("-d", "2", "Number of dimensions", true);
  // cl.addArgument("-f", "-1", "Index for function value", false);
  cl.addArgument("-f", "Ye", "string for the function", false);
  cl.addArgument("-q", "-1", "QuerySize for NGLIterator", false);
  cl.addArgument("-l", "ann", "Neighborhood query library", false);

  cl.addArgument("-k", "-1", "K max", false);
  cl.addArgument("-b", "1.0", "Beta", false);
  cl.addArgument("-g", "0", "KNN", false);
  cl.addArgument("-p", "2.0", "Lp-norm", false);
  cl.addArgument("-r", "1", "Relaxed", false);
  cl.addArgument("-s", "-1", "# of Discretization Steps. Use -1 to disallow discretization.", false);
  cl.addArgument("-o", "output.hdff", "Output file name", false);
  cl.addArgument("-i", "input.txt", "Input meta file name", true);

  bool hasArguments = cl.processArgs(argc, argv);
  if(!hasArguments) {
    fprintf(stderr, "Missing arguments\n");
    fprintf(stderr, "Usage:\n\n");
    cl.showUsage();
    exit(1);
  }

  // int D = cl.getArgInt("-d");
  int Q = cl.getArgInt("-q");
  int A = cl.getArgInt("-a");

  // int F = cl.getArgInt("-f");
  std::string func = cl.getArgString("-f");

  int K = cl.getArgInt("-k");
  if (K==-1)
     K=500;
  std::string input_name = cl.getArgString("-i");
  std::string out_name = cl.getArgString("-o");
  int steps = cl.getArgInt("-s");

  //bool discrete = steps > 0;
  bool relaxed = cl.getArgInt("-r") > 0;
  bool KNN = cl.getArgInt("-g") > 0;
  float beta = cl.getArgFloat("-b");
  float lp = cl.getArgFloat("-p");


  std::vector<float> buffer;
  std::vector<std::string> attrs;
  int N, D;

  if(!loadBin(input_name, func, buffer, N, D, attrs))
    fprintf(stderr, "fail to load file: %s", input_name.c_str());

  fprintf(stderr, "\nAttributes:\n");
  for(size_t i=0; i<attrs.size(); i++)
    fprintf(stderr, "  %s \n ", attrs[i].c_str());

  //set default attribution
  if(A==-1)
    A = D+1;

  // if not set, use last attr
  int F = -1;
  if(F==-1)
    F = A-1;

  // Load data set and edges from files
  int i;

  // Adding colors for terminal output
  Color::Modifier red(Color::FG_RED);
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier blue(Color::FG_BLUE);
  Color::Modifier def(Color::FG_DEFAULT);

  // x = new float[N*D];
  // xy = new float[N*(D+1)];
  // std::vector<std::string> attrs;

  // for(i=0; i<D; i++)
  //   attrs.push_back(std::string("X")+std::to_string(i));
  // attrs.push_back(std::string("f"));
  float *x;
  float *xy;
  x = new float[N*D];
  xy = &buffer[0];
  // xy = new float[N*A];

  // std::vector<std::string> attrs;
  // for(i=0; i<A; i++)
  // {
  //   if (i!=F)
  //     attrs.push_back(std::string("X")+std::to_string(i));
  //   else
  //     attrs.push_back(std::string("f"));
  // }

  t2 = now();
  std::cerr << blue << "Setup and Memory Allocation " << t2-t1 << " s" << def << std::endl;
  t1 = now();

  // std::ifstream in(input_name,std::ios_base::binary);
  // in.read( reinterpret_cast<char *>(&xy[0]), A*N*sizeof(float) );

  std::cout << D << ' ' << N << '\n' << input_name<< ' ';

  for(unsigned i = 0 ; i < N ; ++i )
  {
      for (unsigned j = 0; j<D; ++j)
      {
        x[i*D+j] = xy[i*A+j];
      }
  }

  // in.close();
  //////////////////

  HDData data;
  data.size(N);
  data.dim(D);
  data.attr(A);
  data.func(F);
  data.attributes(attrs);
  data.data(xy);


  t2 = now();
  std::cerr << blue << "loading Data " << t2 - t1 << " s" << def << std::endl;
  t1 = now();

  //////////////// Compute Extrema Graph ////////////////
  ExtremumGraphExt eg;
  Flags *flags = NULL;

#ifdef ENABLE_STREAMING
  SearchIndex *index = NULL;
  std::string library = cl.getArgString("-l");
  fprintf(stderr, "using %s library\n", library.c_str());
  if (library.compare("ANN") == 0 || library.compare("ann") == 0) {
   index = new ANNSearchIndex(0);
  }
  else if (library.compare("FAISS") == 0 || library.compare("faiss") == 0) {
#ifdef ENABLE_FAISS
   index = new FAISSSearchIndex();
#else
    std::cerr << "Exit: Did't enable FAISS library\n" << std::endl;
    exit(0);
#endif
  }
  else if (library.compare("FLANN") == 0 || library.compare("flann") == 0) {
#ifdef ENABLE_FLANN
   index = new FLANNSearchIndex();
#else
   std::cerr << "Exit: Did't enable FLANN library\n" << std::endl;
   exit(0);
#endif
  }

  NGLIterator it(x, N, D, K, relaxed, beta, lp, steps, Q, index);
  eg.initialize(&data, flags, it, true, 10, ExtremumGraphExt::ComputeMode::HISTOGRAM);

  t2 = now();
  std::cerr << blue << "Building Extremum Graph with streaming NGLIterator: " << t2-t1 << " s" << def << std::endl;
  t1 = now();

#else ////////// directly compute the empty region graph using ngl //////////////
  IndexType *indices;
  int numEdges;

  // NGLIterator it(x, N, D, K, relaxed, beta, lp, steps, Q);
  //eg.initialize(&data, it, true, 10, ExtremumGraphExt::ComputeMode::HISTOGRAM);

  Geometry<float>::init(D);   // Initialize NGL for 2-dimensional points

  ANNPointSet<float> P(&x[0], N);  // Initialize Point set using ANN (computes a kd-tree)

  NGLParams<float> params;
  params.iparam0 = K;//150;      // Initialize parameters
  params.param1 = beta; //1.5; //beta
                            // Only computes Gabriel graph from the KMAX nearest
                            // neighbor graph, computed using ANN
  getGabrielGraph(P, &indices, numEdges, params);   // Get graph
  std::cout<<"numEdges:"<<numEdges<<std::endl;

  Neighborhood edges(indices, numEdges);
  eg.initialize(&data, flags, &edges, true, 10, ExtremumGraphExt::ComputeMode::HISTOGRAM);

  t2 = now();
  std::cerr << blue << "Building Extremum Graph with NGL: " << t2-t1 << " s" << def << std::endl;
  t1 = now();
#endif

  HDFileFormat::DataBlockHandle mc;
  mc.idString("TDA");
  eg.save(mc);
  HDFileFormat::DatasetHandle dataset;
  dataset.add(mc);

  HDFileFormat::DataCollectionHandle group(out_name);

  group.add(dataset);
  group.write();

  t2 = now();
  std::cerr << blue << "Output " << t2-t1 << " s" << def << std::endl;
  t1 = now();

  // // Free memory
  delete [] x;
  // delete [] xy;

  t2 = now();
  std::cerr << blue << "Clean-up " << t2-t1 << " s" << def << std::endl;

   return 0;
}
