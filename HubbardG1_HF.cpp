#include <iostream>
#include <eigen3/Eigen/Core>
#include <complex>
#include <fstream>
#include <cmath>

constexpr std::complex<double> imu{0,1.0};

int Npart=5;
int Nsites=10;
int Nt=100000;
double dt=0.01;
double U=5;
int Outrat=1;

Eigen::MatrixXcd G1(Nsites,Nsites);
Eigen::MatrixXcd d1G1(Nsites,Nsites);
Eigen::MatrixXcd d2G1(Nsites,Nsites);
Eigen::MatrixXcd d3G1(Nsites,Nsites);
Eigen::MatrixXcd d4G1(Nsites,Nsites);
Eigen::MatrixXcd tempState(Nsites,Nsites);
Eigen::MatrixXcd T(Nsites,Nsites);
Eigen::MatrixXcd W(Nsites,Nsites);
Eigen::MatrixXcd H(Nsites,Nsites);
Eigen::MatrixXcd OutputMatMul(Nsites,Nsites);

int init_G1();
int buildW();
int buildT();
int calcDeriv(Eigen::MatrixXcd& Input, Eigen::MatrixXcd& Output);
int leftSparseMatMul(Eigen::MatrixXcd& A, Eigen::MatrixXcd& B, Eigen::MatrixXcd& C);
int rightSparseMatMul(Eigen::MatrixXcd& A, Eigen::MatrixXcd& B, Eigen::MatrixXcd& C);
int calcObs();

int main(){
    double t;

    T.setZero();
    W.setZero();
    H.setZero();
    // initialise G1_0
    init_G1();

    // build kinectic Operator
    buildT();

    // Initial values
    std::cout << "#Perc\t" << "time\t" << "N\t" << "Ekin\t" << "Epot\t" << "Eges" << std::endl;
    std::cout << round(0./(Nt-1)*1000.)/10. << "%\t" << 0.*dt << "\t";
    calcObs();
    // do Nt time steps and calc obs RK4
    for(int i=1;i<Nt;i++){
        t=i;
        buildW();
        H=T+W;

        calcDeriv(G1,d1G1);
        tempState=G1+(0.5*dt)*d1G1;
        calcDeriv(tempState,d2G1);
        tempState=G1+(0.5*dt)*d2G1;
        calcDeriv(tempState,d3G1);
        tempState=G1+(1.0*dt)*d3G1;
        calcDeriv(tempState,d4G1);

        G1+=dt*(d1G1+2.*(d2G1+d3G1)+d4G1)/6.;


        if(int(t)%Outrat==0){
            std::cout << round(t/(Nt-1)*1000.)/10. << "%\t" << t*dt << "\t";
            calcObs();
        }
    }
    return 0;
}

int init_G1(){
    // initializes G1 with Npart Particles at the first two sites
    G1.setZero();
    for(int i=0;i<Npart;i++){
        G1(i,i)=imu;
    }
    return 0;
    //G1 << real(Eigen::MatrixXcd::Random(Nsites,Nsites);
    return 0;
}

int buildW(){
    // build Interaction Potential for specific time
    for(int i=0;i<Nsites;i++){
        W(i,i)=-imu*U*G1(i,i);
    }
    return 0;
}

int buildT(){
    // build Kinetic Operator
    for(int i=1;i<Nsites-1;i++){
        T(i,i-1)=-1;
        T(i,i+1)=-1;
    }
    T(0,1)=-1;T(Nsites-1,Nsites-2)=-1;
    return 0;
}

int calcDeriv(Eigen::MatrixXcd& Input, Eigen::MatrixXcd& Output){
    // calculate the teporal derivative
    leftSparseMatMul(H,Input,OutputMatMul);
    Output=OutputMatMul;

    rightSparseMatMul(Input,H,OutputMatMul);
    Output-=OutputMatMul;
    Output*=-imu;
    return 0;
}

int leftSparseMatMul(Eigen::MatrixXcd& A, Eigen::MatrixXcd& B, Eigen::MatrixXcd& C){
    // calculates the matrix product of a sparse Matrix A and a Matrix B: A*B
    // need to specify number of diagonals that are !=0 in sparse matrix. Hubbard its 3 (k=-1,0,1)
    C.setZero();

    for (int j=0;j<Nsites;j++){
        for(int k=-1;k<2;k++){
            for(int i=1;i<Nsites-1;i++){
                C(i,j)+=A(i,i+k)*B(i+k,j);
            }
        }
        for(int k=0;k<2;k++){
            C(0,j)+=A(0,k)*B(k,j);
            C(Nsites-1,j)+=A(Nsites-1,Nsites-1-k)*B(Nsites-1-k,j);
        }
    }
    return 0;
}

int rightSparseMatMul(Eigen::MatrixXcd& A, Eigen::MatrixXcd& B, Eigen::MatrixXcd& C){
    // calculates the matrix product of a Matrix A and a sparse Matrix B: A*B
    // need to specify number of diagonals that are !=0 in sparse matrix. Hubbard its 3 (k=-1,0,1)
    C.setZero();

    for (int i=0;i<Nsites;i++){
        for(int k=-1;k<2;k++){
            for(int j=1;j<Nsites-1;j++){
                C(i,j)+=A(i,j+k)*B(j+k,j);
            }
        }
        for(int k=0;k<2;k++){
            C(i,0)+=A(i,k)*B(k,0);
            C(i,Nsites-1)+=A(i,Nsites-1-k)*B(Nsites-1-k,Nsites-1);
        }
    }
    return 0;
}

int calcObs(){
    // Calculates Observables and prints them to console
    double N=0;
    std::complex<double> Ekin=0;
    std::complex<double> Eint=0;
    double Eges=0;
    leftSparseMatMul(T,G1,OutputMatMul);
    //std::cout << G1 << std::endl << std::endl;
    for(int i=0;i<Nsites;i++){
        N+=imag(G1(i,i));
        Ekin+=OutputMatMul(i,i);
        Eint+=U*G1(i,i)*G1(i,i);
    }
    Eint/=-2.;
    Eges=imag(Ekin)+real(Eint);
    //std::cout << std::endl << OutputMatMul << std::endl;
    std::cout << N << "\t" << imag(Ekin) << "\t" << real(Eint) << "\t" << Eges << std::endl;
    return 0;
}