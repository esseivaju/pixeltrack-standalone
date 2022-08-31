#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "test_common.h"

using namespace Eigen;

__host__ __device__ void eigenValues(Matrix3d *m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret) {
#if TEST_DEBUG
  printf("Matrix(0,0): %f\n", (*m)(0, 0));
  printf("Matrix(1,1): %f\n", (*m)(1, 1));
  printf("Matrix(2,2): %f\n", (*m)(2, 2));
#endif
  SelfAdjointEigenSolver<Matrix3d> es;
  es.computeDirect(*m);
  (*ret) = es.eigenvalues();
  return;
}

__global__ void kernel(Matrix3d *m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret) {
  eigenValues(m, ret);
}

void testEigenvalues() {
  std::cout << "TEST EIGENVALUES" << std::endl;
  Matrix3d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix3d *m_gpu = nullptr;
  Matrix3d *mgpudebug = new Matrix3d();
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret =
      new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret1 =
      new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret_gpu = nullptr;
  eigenValues(&m, ret);
#if TEST_DEBUG
  std::cout << "Generated Matrix M 3x3:\n" << m << std::endl;
  std::cout << "The eigenvalues of M are:" << std::endl << (*ret) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  cudaMalloc((void **)&m_gpu, sizeof(Matrix3d));
  cudaMalloc((void **)&ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType));
  cudaMemcpy(m_gpu, &m, sizeof(Matrix3d), cudaMemcpyHostToDevice);

  kernel<<<1, 1>>>(m_gpu, ret_gpu);
  cudaDeviceSynchronize();

  cudaMemcpy(mgpudebug, m_gpu, sizeof(Matrix3d), cudaMemcpyDeviceToHost);
  cudaMemcpy(
      ret1, ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType), cudaMemcpyDeviceToHost);
#if TEST_DEBUG
  std::cout << "GPU Generated Matrix M 3x3:\n" << (*mgpudebug) << std::endl;
  std::cout << "GPU The eigenvalues of M are:" << std::endl << (*ret1) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  assert(isEqualFuzzy(*ret, *ret1));
}

int main(int argc, char *argv[]) {

  testEigenvalues();

  return 0;
}