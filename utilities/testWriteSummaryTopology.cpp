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
#include <HDData.h>

#include <ngl.h>
using namespace ngl;

#include <ExtremumGraph.h>
#include <Neighborhood.h>
#include <NeighborhoodIterator.h>
#include <DataCollectionHandle.h>
#include <DataBlockHandle.h>

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

double Ackley(float *x, int d)
{
    double sum = 0;
    double theta1;
    double theta2;

    for (int i = 0; i < d - 1; i++)
    {
        theta1 = 6 * x[i] - 3;
        theta2 = 6 * x[i + 1] - 3;
        sum -= exp(-0.2) * sqrt(pow(theta1, 2) + pow(theta2, 2)) + 3 * (cos(2 * theta1) + sin(2 * theta2));
        //sum+= 20*exp(-0.2*sqrt(0.5*(pow(x[i],2)+pow(x[i+1],2))))-exp(0.5*(cos(2*3.1415926*x[i])+cos(2*3.1415926*x[i])))+20;
    }
    return sum;
}

double Ackleysfu(float *x, int d)
{

    double sum1 = 0;
    double sum2 = 0;

    //double theta1;
    //double theta2;

    double a = 20;
    double b = 0.2;
    double c = 2*3.1415926536;

    for (int i = 0; i < d; i++)
    {
        //theta1 = 6 * x[i] - 3;
        //theta2 = 6 * x[i + 1] - 3;
        //sum -= exp(-0.2) * sqrt(pow(theta1, 2) + pow(theta2, 2)) + 3 * (cos(2 * theta1) + sin(2 * theta2));
        //sum+= 20*exp(-0.2*sqrt(0.5*(pow(x[i],2)+pow(x[i+1],2))))-exp(0.5*(cos(2*3.1415926*x[i])+cos(2*3.1415926*x[i])))+20;
      sum1+=pow((x[i]-0.5)*32,2);
      sum2+=cos(c*(x[i]-0.5)*32);
    }

    return -a*exp(-b*sqrt(sum1/d))-exp(sum2/d)+a+exp(1);
    //return sum;
}


int main(int argc, char **argv)
{
  timestamp t1 = now();
  timestamp t2;
  CommandLine cl;
  cl.addArgument("-d", "2", "Number of dimensions", false);
  cl.addArgument("-c", "10000", "Number of points", false);
  cl.addArgument("-a", "-1", "Number of attributes", false);
  cl.addArgument("-f", "-1", "Index for function value", false);
  cl.addArgument("-q", "-1", "Query Size for NGLIterator", false);

  cl.addArgument("-k", "500", "K max", false);
  cl.addArgument("-b", "1.0", "Beta", false);
  cl.addArgument("-p", "2.0", "Lp-norm", false);
  cl.addArgument("-r", "1", "Relaxed", false);
  cl.addArgument("-s", "-1", "# of Discretization Steps. Use -1 to disallow discretization.", false);
  cl.addArgument("-o", "summaryTopologyTest.hdff", "Output file name", false);
  bool hasArguments = cl.processArgs(argc, argv);
  if(!hasArguments) {
    fprintf(stderr, "Missing arguments\n");
    fprintf(stderr, "Usage:\n\n");
    cl.showUsage();
    exit(1);
  }

  int D = cl.getArgInt("-d");
  int N = cl.getArgInt("-c");
  int Q = cl.getArgInt("-q");
  // total attrs and index for function value

  int A = cl.getArgInt("-a");
  if(A==-1)
    A = D+1;

  int F = cl.getArgInt("-f");
  // if not set, use last attr
  if(F==-1)
    F = A-1;

  int K = cl.getArgInt("-k");
  std::string out_name = cl.getArgString("-o");
  int steps = cl.getArgInt("-s");

  //bool discrete = steps > 0;

  bool relaxed = cl.getArgInt("-r") > 0;
  float beta = cl.getArgFloat("-b");
  float lp = cl.getArgFloat("-p");

  // Load data set and edges from files
  float *x;
  float *xy;

  //int i, d, k;
  int i, d;


  // Adding colors for terminal output
  Color::Modifier red(Color::FG_RED);
  Color::Modifier green(Color::FG_GREEN);
  Color::Modifier blue(Color::FG_BLUE);
  Color::Modifier def(Color::FG_DEFAULT);

  x = new float[N*D];
  xy = new float[N*A];

  std::vector<std::string> attrs;
  for(i=0; i<A; i++)
  {
    if (i!=F)
      attrs.push_back(std::string("X")+std::to_string(i));
    else
      attrs.push_back(std::string("f"));
  }

  t2 = now();
  std::cerr << blue << "Setup and Memory Allocation " << t2-t1 << " s" << def << std::endl;
  t1 = now();

  srand(0);
//   std::ofstream ofs;
//   ofs.open("points.txt");
  for (i = 0; i < N; i++)
  {
      for (d = 0; d < D; d++)
      {
          xy[i*A+d] = x[i * D + d] = (float)rand() / RAND_MAX;
          //xy[i*(D+1)+d] = x[i * D + d] = ((float)rand() / RAND_MAX-0.5)*8;

          //ofs << x[i * D + d] << " ";
      }
      //ofs << std::endl;
  }
  //ofs.close();

  for(i = 0; i < N; i++) {
      //xy[i*(D+1)+D] = Ackley(x+(i*D), D);
      for(int ind = D; ind<A; ind++)
        xy[i*A+ind] = -Ackley(x+(i*D), D)-ind;
  }

  HDData data;
  data.size(N);
  data.dim(D);
  data.attr(A);
  data.func(F);
  data.attributes(attrs);
  data.data(xy);

  t2 = now();
  std::cerr << blue << "Generating Data " << t2 - t1 << " s" << def << std::endl;
  t1 = now();

 //////////////////////////////// Replace this block with an EdgeIterator
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

  ////////////////////////////////


  ExtremumGraphExt eg;
  Flags *flags = NULL;

  eg.initialize(&data, flags, &edges, true, 10, ExtremumGraphExt::ComputeMode::HISTOGRAM);

  t2 = now();
  std::cerr << blue << "Building Extremum Graph with NeighborhoodIterator " << t2-t1 << " s" << def << std::endl;
  t1 = now();

  HDFileFormat::DataBlockHandle mc;
  mc.idString("TDA");
  eg.save(mc);
  HDFileFormat::DatasetHandle dataset;
  dataset.add(mc);

  HDFileFormat::DataCollectionHandle group(out_name);
  //HDFileFormat::DataCollectionHandle group("topo_5d_10M_k250.hdff");
  group.add(dataset);
  group.write();

  t2 = now();
  std::cerr << blue << "Output " << t2-t1 << " s" << def << std::endl;
  t1 = now();

  // Free memory
  delete [] x;
  delete [] xy;

  t2 = now();
  std::cerr << blue << "Clean-up " << t2-t1 << " s" << def << std::endl;

  return 0;
}
