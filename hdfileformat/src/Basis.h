#ifndef BASIS_H
#define BASIS_H

#include <iostream>

namespace HDFileFormat{

class Basis
{
public:
  Basis()
  {
    rows = cols = 0;
  }

  bool empty()
  {
    return !(rows && cols);
  }

  void resize(int row, int col)
  {
    rows = row;
    cols = col;
    coeffColMajor.resize(rows*cols);
  }

  Basis(int rows_, int cols_, double* colMajor = NULL)
    :rows(rows_), cols(cols_)
  {
    //allocate its own memory
    coeffColMajor.resize(rows*cols);
    if(colMajor)
      std::copy(colMajor, colMajor+rows*cols, coeffColMajor.begin());
  }

  double Coeff(int row, int col) const
  {
    int index = col*cols+row;
    if(index< int(coeffColMajor.size()) )
      return coeffColMajor[index];
    else
      return zero;
  }
  double& Coeff(int row, int col)
  {
    //rowId + colId * m_storage.rows()
    int index = col*rows + row;
    if(index< int(coeffColMajor.size()) )
      return coeffColMajor[index];
    else
      return zero;
  }
  //! coeff of the matrix stored in column major
  std::vector<double> coeffColMajor;
  //! number of cols (basis dimension)
  int cols;
  //! number of original dimension
  int rows;

  friend std::ostream& operator<<(std::ostream& os, const Basis& basis);

private:
  //init to zero in the cpp file
  static double zero;
};

//! support cout<< for the basis
inline std::ostream& operator<<(std::ostream& os, const Basis& basis)
{
  if(basis.cols==0)
    os << "[Empty Basis]";
  os << " [ ";
  for(int i=0; i<basis.cols; i++)
  {
    for(int j=0; j<basis.rows; j++)
      os << basis.Coeff(j,i) << " ";
    if(i < basis.cols-1)
      os << ", ";
  }
  os << "] ";
  return os;
}

}//namespace

#endif
