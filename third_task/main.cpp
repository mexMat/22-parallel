#include <iostream>
#include <cmath>
#include <mpi.h>

struct Config {
    int size_x, size_y;

    double h_1, h_2;

    double x_left, x_right, y_bot, y_top;

    double alpha_L, alpha_R, alpha_B, alpha_T;

    double (*u_func)(double, double);

    double (*k_func)(double, double);

    double (*q_func)(double, double);

    double (*f_func)(double, double);

    double (*edge_top)(double, double);

    double (*edge_bot)(double, double);

    double (*edge_right)(double, double);

    double (*edge_left)(double, double);

    Config() = default;

    Config(int size_x, int size_y,
           double x_left, double x_right,
           double y_bot, double y_top,
           double alpha_L, double alpha_R,
           double alpha_B, double alpha_T,
           double (*u_func)(double, double),
           double (*k_func)(double, double),
           double (*q_func)(double, double),
           double (*f_func)(double, double),
           double (*edge_top)(double, double),
           double (*edge_bot)(double, double),
           double (*edge_left)(double, double),
           double (*edge_right)(double, double)) {
        this->size_x = size_x;
        this->size_y = size_y;
        this->h_1 = (double) (x_right - x_left) / (double) (size_x - 1);
        this->h_2 = (double) (y_top - y_bot) / (double) (size_y - 1);
        this->x_left = x_left;
        this->x_right = x_right;
        this->y_top = y_top;
        this->y_bot = y_bot;
        this->alpha_L = alpha_L;
        this->alpha_B = alpha_B;
        this->alpha_R = alpha_R;
        this->alpha_T = alpha_T;
        this->u_func = u_func;
        this->k_func = k_func;
        this->q_func = q_func;
        this->f_func = f_func;
        this->edge_bot = edge_bot;
        this->edge_top = edge_top;
        this->edge_left = edge_left;
        this->edge_right = edge_right;
    }
};

class Matrix {
public:
    int x_size, y_size, k_size;
    int size;
    double *matrix;

    Matrix() = default;

    Matrix(Matrix const &) = default;

    Matrix(int x_size, int y_size, int k_size) {
        this->x_size = x_size;
        this->y_size = y_size;
        this->k_size = k_size;
        this->size = x_size * y_size * k_size;
        this->matrix = new double[size];
    }

    int getSizeX() const {
        return this->x_size;
    }

    int getSizeY() const {
        return this->y_size;
    }

    int getSizeK() const {
        return this->k_size;
    }

    static Matrix &sub(Matrix &a, Matrix &b, Matrix &result) {
        int M = a.getSizeX();
        int N = a.getSizeY();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                result(i, j, 0) = a(i, j, 0) - b(i, j, 0);
        return result;
    }

    static Matrix &copy(Matrix &a, Matrix &result) {
        int M = a.getSizeX();
        int N = a.getSizeY();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                result(i, j, 0) = a(i, j, 0);
    }

    static Matrix &mul(Matrix &A, Matrix &b, Matrix &result) {
        int M = A.getSizeX();
        int N = A.getSizeY();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                // bottom points
                if (i == 0) {
                    //left bottom
                    if (j == 0) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j + 1, 0) +
                                A(i, j, 1) * b(i + 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    //right bottom
                    if (j == M - 1) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j - 1, 0) +
                                A(i, j, 1) * b(i + 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    //edge
                    result(i, j, 0) =
                            A(i, j, 0) * b(i + 1, j, 0) +
                            A(i, j, 1) * b(i, j - 1, 0) +
                            A(i, j, 2) * b(i, j + 1, 0) +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                // top points
                if (i == N - 1) {
                    //right top
                    if (j == M - 1) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j - 1, 0) +
                                A(i, j, 1) * b(i - 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    //left top
                    if (j == 0) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j + 1, 0) +
                                A(i, j, 1) * b(i - 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    //edge
                    result(i, j, 0) =
                            A(i, j, 0) * b(i - 1, j, 0) +
                            A(i, j, 1) * b(i, j - 1, 0) +
                            A(i, j, 2) * b(i, j + 1, 0) +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                //left points
                if (j == 0) {
                    //edge
                    result(i, j, 0) =
                            A(i, j, 0) * b(i, j + 1, 0) +
                            A(i, j, 1) * b(i + 1, j, 0) +
                            A(i, j, 2) * b(i - 1, j, 0) +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                //right points
                if (j == M - 1) {
                    //edge
                    result(i, j, 0) =
                            A(i, j, 0) * b(i, j - 1, 0) +
                            A(i, j, 1) * b(i + 1, j, 0) +
                            A(i, j, 2) * b(i - 1, j, 0) +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                // inner points
                result(i, j, 0) =
                        A(i, j, 0) * b(i, j - 1, 0) +
                        A(i, j, 1) * b(i, j + 1, 0) +
                        A(i, j, 2) * b(i - 1, j, 0) +
                        A(i, j, 3) * b(i + 1, j, 0) +
                        A(i, j, 4) * b(i, j, 0);
            }
        return result;
    }

    static Matrix &coef(Matrix &A, double alpha) {
        int M = A.getSizeX();
        int N = A.getSizeY();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j)
                A(i, j, 0) = alpha * A(i, j, 0);
    }

    double getItem(int i, int j, int k) const {
        return matrix[i * x_size * k_size + j * k_size + k];
    }

    double &operator()(int i, int j, int k) {
        return matrix[i * x_size * k_size + j * k_size + k];
    }

    void print_plain() const {
        for (int i = 0; i < size; ++i)
            std::cout << this->matrix[i] << " ";
        std::cout << std::endl;
        for (int i = 0; i < size; ++i)
            std::cout << i << " ";
    };

    void print() const {
        for (int i = 0; i < getSizeX(); ++i)
            for (int j = 0; j < getSizeY(); ++j) {
                for (int k = 0; k < getSizeK(); ++k)
                    std::cout << getItem(i, j, k) << " ";
                std::cout << "(" << i << ", " << j << ")\n";
            }
    }
};

// N - number of rows
// M - number of columns
class PuassonEquation {
    static double dot(Matrix &a, Matrix &b, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        double h1 = config->h_1;
        double h2 = config->h_2;
        double p1, p2;
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                sum += h1 * h2 * a(i, j, 0) * b(i, j, 0);
//                if ((i == 0) or (i == N - 1))
//                    p2 = 0.5;
//                else
//                    p2 = 1.;
//                if ((j == 0) or (j == N - 1))
//                    p1 = 0.5;
//                else
//                    p1 = 1.;
//                sum += h1 * h2 * p1 * p2 * a(i, j, 0) * b(i, j, 0);
            }
        return sum;
    }

    static double norm_c(Matrix &a, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        double max = a(0, 0, 0);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                if (fabs(a(i, j, 0)) > max)
                    max = a(i, j, 0);
            }
        return fabs(max);
    }

    static double norm(Matrix &a, Config *config) {
        return dot(a, a, config);
    }

    static Matrix filling(Matrix &matrix, Matrix &f_vector, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        double h1 = config->h_1;
        double h2 = config->h_2;
        double h1_2 = h1 * h1, h2_2 = h2 * h2;
        double x_left = config->x_left, x_right = config->x_right;
        double y_bot = config->y_bot, y_top = config->y_top;
        double xi, yj;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                xi = x_left + j * h1;
                yj = y_bot + i * h2;
                // bottom points
                if (i == 0) {
                    //left bottom
                    if (j == 0) {
                        matrix(i, j, 0) =
                                -(2. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                        matrix(i, j, 1) =
                                -(2. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_L / h1 +
                                2. * config->alpha_B / h2;
                        f_vector(i, j, 0) =
                                config->f_func(xi, yj) +
                                (2. / h1 + 2. / h2) *
                                (h1 * config->edge_bot(xi, yj) +
                                 h2 * config->edge_left(xi, yj)) / (h1 + h2);
                        continue;
                    }
                    //right bottom
                    if (j == M - 1) {
                        matrix(i, j, 0) =
                                -(2. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                        matrix(i, j, 1) =
                                -(2. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_R / h1 +
                                2. * config->alpha_B / h2;
                        f_vector(i, j, 0) =
                                config->f_func(xi, yj) +
                                (2. / h1 + 2. / h2) *
                                (h1 * config->edge_bot(xi, yj) +
                                 h2 * config->edge_right(xi, yj)) / (h1 + h2);
                        continue;
                    }
                    //edge
                    matrix(i, j, 0) =
                            -(2. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 1) =
                            -(1. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 2) =
                            -(1. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_B / h2;
                    f_vector(i, j, 0) =
                            config->f_func(xi, yj) + (2. / h2) * config->edge_bot(xi, yj);
                    continue;
                }
                // top points
                if (i == N - 1) {
                    //right top
                    if (j == M - 1) {
                        matrix(i, j, 0) =
                                -(2. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                        matrix(i, j, 1) =
                                -(2. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_R / h1 +
                                2. * config->alpha_T / h2;
                        f_vector(i, j, 0) =
                                config->f_func(xi, yj) +
                                (2. / h1 + 2. / h2) *
                                (h1 * config->edge_top(xi, yj) +
                                 h2 * config->edge_right(xi, yj)) / (h1 + h2);
                        continue;
                    }
                    //left top
                    if (j == 0) {
                        matrix(i, j, 0) =
                                -(2. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                        matrix(i, j, 1) =
                                -(2. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_L / h1 +
                                2. * config->alpha_T / h2;
                        f_vector(i, j, 0) =
                                config->f_func(xi, yj) +
                                (2. / h1 + 2. / h2) *
                                (h1 * config->edge_top(xi, yj) +
                                 h2 * config->edge_left(xi, yj)) / (h1 + h2);
                        continue;
                    }
                    // EDGE
                    matrix(i, j, 0) =
                            -(2. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 1) =
                            -(1. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 2) =
                            -(1. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_T / h2;
                    f_vector(i, j, 0) =
                            config->f_func(xi, yj) + (2. / h2) * config->edge_top(xi, yj);
                    continue;
                }
                //left points
                if (j == 0) {
                    //edge
                    matrix(i, j, 0) =
                            -(2. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 1) =
                            -(1. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 2) =
                            -(1. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_L / h1;
                    f_vector(i, j, 0) =
                            config->f_func(xi, yj) + (2. / h1) * config->edge_left(xi, yj);
                    continue;
                }
                //right points
                if (j == M - 1) {
                    //edge
                    matrix(i, j, 0) =
                            -(2. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 1) =
                            -(1. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 2) =
                            -(1. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_R / h1;
                    f_vector(i, j, 0) =
                            config->f_func(xi, yj) + (2. / h1) * config->edge_right(xi, yj);
                    continue;
                }
                // inner points
                {
                    matrix(i, j, 0) =
                            -1. / (h1_2) * config->k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 1) =
                            -1. / (h1_2) * config->k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 2) =
                            -1. / (h2_2) * config->k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 3) =
                            -1. / (h2_2) * config->k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 4) =
                            -matrix(i, j, 0) - matrix(i, j, 1) -
                            matrix(i, j, 2) - matrix(i, j, 3) +
                            config->q_func(xi, yj);
                    f_vector(i, j, 0) = config->f_func(xi, yj);
                }
            }
    }

    static Matrix optimize(Matrix &matrix, Matrix &f_vector, Matrix &w_vector,
                           Matrix &r_vector, Matrix &Ar_vector, Matrix &true_solution, Config *config) {
        double tau_k = 0.;
        double eps = 10.;
        int iter = 0;
        while (eps > 1e-06) {
            Matrix::mul(matrix, w_vector, Ar_vector);
            Matrix::sub(Ar_vector, f_vector, r_vector);
            Matrix::mul(matrix, r_vector, Ar_vector);
            tau_k = dot(Ar_vector, r_vector, config);
            tau_k = tau_k / dot(Ar_vector, Ar_vector, config);
            Matrix::coef(r_vector, tau_k);
            Matrix::sub(w_vector, r_vector, Ar_vector);
            Matrix::sub(Ar_vector, w_vector, r_vector);
            eps = norm_c(r_vector, config);
//            eps = sqrt(norm(r_vector, config));
            std::cout << "iter: " << iter << "; eps: " << eps << std::endl;
            iter++;
            Matrix::copy(Ar_vector, w_vector);
        }
    }

    static void set_true_solution(Matrix &a, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        double h1 = config->h_1;
        double h2 = config->h_2;
        double h1_2 = h1 * h1, h2_2 = h2 * h2;
        double x_left = config->x_left, x_right = config->x_right;
        double y_bot = config->y_bot, y_top = config->y_top;
        double xi, yj;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                xi = x_left + j * h1;
                yj = y_bot + i * h2;
                a(i, j, 0) = config->u_func(xi, yj);
            }
    }

public:
    static Matrix solve(Config *config) {
        Matrix matrix(config->size_x, config->size_y, 5);
        Matrix f_vector(config->size_x, config->size_y, 1);
        Matrix w_vector(config->size_x, config->size_y, 1);
        Matrix r_vector(config->size_x, config->size_y, 1);
        Matrix Ar_vector(config->size_x, config->size_y, 1);
        Matrix true_solution(config->size_x, config->size_y, 1);
        set_true_solution(true_solution, config);
        filling(matrix, f_vector, config);
        optimize(matrix, f_vector, w_vector, r_vector, Ar_vector, true_solution, config);
        Matrix::sub(true_solution, w_vector, r_vector);
        std::cout << "norm abs: " << sqrt(norm(r_vector, config)) << std::endl;
        std::cout << "max abs: " << norm_c(r_vector, config) << std::endl;
        Matrix::mul(matrix, true_solution, Ar_vector);
        Matrix::sub(Ar_vector, f_vector, r_vector);
        std::cout<<"!!!!!!!!!!!!!!!!!: "<<std::endl;
        r_vector.print();
        std::cout<<"?????????????????: "<<std::endl;
        std::cout << "true residual abs: " << sqrt(norm(r_vector, config)) << std::endl;
        Matrix::mul(matrix, w_vector, Ar_vector);
        Matrix::sub(Ar_vector, f_vector, r_vector);
        std::cout << "residual abs: " << sqrt(norm(r_vector, config)) << std::endl;
    }
};


int main(int argc, char *argv[]) {

    int rank, size = 9;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2];
    [](int size, int *dims) {
        int i;
        for (i = 1; i < (size >> 1); ++i) {
            if ((size / i <= i) and (size % i == 0)) {
                break;
            }
        }
        dims[0] = i;
        dims[1] = size / i;
    }(size, dims);

    std::cout << dims[0] << " " << dims[1] << std::endl;
    //my config
    Config config(
            atoi(argv[1]), atoi(argv[2]),
            0., 4.,
            0., 3.,
            1., 1.,
            0., 0.,
            [](double x, double y) -> double { return sqrt(4 + x * y); },
            [](double x, double y) -> double { return 4 + x + y; },
            [](double x, double y) -> double { return x + y; },
            [](double x, double y) -> double {
                return x * x * (x + y + 4.) / (4. * pow(x * y + 4., 1.5)) +
                       y * y * (x + y + 4.) / (4. * pow(x * y + 4, 1.5)) -
                       x / (2. * pow(x * y + 4., 0.5)) -
                       y / (2. * pow(x * y + 4., 0.5)) +
                       (x + y) * pow(x * y + 4., 0.5);
            },
            [](double x, double y) -> double {
                return x * (4 + x + y) / (2 * pow(4 + x * y, 0.5));
            },
            [](double x, double y) -> double {
                return -x * (4 + x + y) / (2 * pow(4 + x * y, 0.5));
            },
            [](double x, double y) -> double {
                return -y * (4 + x + y) / (2 * pow(4 + x * y, 0.5))
                       + pow(4 + x * y, 0.5);
            },
            [](double x, double y) -> double {
                return y * (4 + x + y) / (2 * pow(4 + x * y, 0.5))
                       + pow(4 + x * y, 0.5);
            }
    );

    //config ramil

        Config config_ramil(
                atoi(argv[1]), atoi(argv[2]),
                -2., 3.,
                -1., 4.,
                1., 1.,
                1., 1.,
                [](double x, double y) -> double { return 2. / (1. + x * x + y * y); },
                [](double x, double y) -> double { return 4 + x; },
                [](double x, double y) -> double { return 1; },
                [](double x, double y) -> double {
                    return (-4. * x - 32.) / pow((1. + x * x + y * y), 2) +
                           16. * (x + 4) / pow((1 + x * x + y * y), 3) +
                           2. / (1 + x * x + y * y);
                },
                [](double x, double y) -> double {
                    return 2. * (-8 * y - 2. * x * y + x * x + y * y + 1) / pow((1 + x * x + y * y), 2.);
                },
                [](double x, double y) -> double {
                    return 2. * (8 * y + 2. * x * y + x * x + y * y + 1) / pow((1 + x * x + y * y), 2.);
                },
                [](double x, double y) -> double {
                    return 2. * (3. * x * x + y * y + 8. * x + 1) / pow((1 + x * x + y * y), 2.);
                },
                [](double x, double y) -> double {
                    return 2. * (-x * x + y * y - 8. * x + 1) / pow((1 + x * x + y * y), 2.);
                }
        );

    //config kirill
    {
        Config config_kirill(
                atoi(argv[1]), atoi(argv[2]),
                -2., 3.,
                -1., 4.,
                0., 0.,
                0., 0.,
                [](double x, double y) -> double { return 2.0 / (1.0 + x * x + y * y); },
                [](double x, double y) -> double { return 1.0 + (x + y) * (x + y); },
                [](double x, double y) -> double { return 1.0; },
                [](double x, double y) -> double {
                    return 2.0 / (1. + x * x + y * y) -
                           ((4.0 * (x * x * x * x + 4.0 * x * x * x * y - 4.0 * x * (y * y * y + y) -
                                    (y * y + 1.0) * (y * y + 1.0)) /
                             ((1. + x * x + y * y) * (1. + x * x + y * y) * (1. + x * x + y * y))) -
                            (4.0 * (x * x * x * x + 4.0 * x * x * x * y + 2.0 * x * x - 4.0 * x * y * (y * y - 1.0) -
                                    y * y * y * y + 1.0) /
                             ((1. + x * x + y * y) * (1. + x * x + y * y) * (1. + x * x + y * y))));
                },
                [](double x, double y) -> double {
                    return -y * 4.0 * (1.0 + (x + y) * (x + y)) / ((1. + x * x + y * y) * (1. + x * x + y * y));
                },
                [](double x, double y) -> double {
                    return y * 4.0 * (1.0 + (x + y) * (x + y)) / ((1. + x * x + y * y) * (1. + x * x + y * y));
                },
                [](double x, double y) -> double {
                    return x * 4.0 * (1.0 + (x + y) * (x + y)) / ((1. + x * x + y * y) * (1. + x * x + y * y));
                },
                [](double x, double y) -> double {
                    return -x * 4.0 * (1.0 + (x + y) * (x + y)) / ((1. + x * x + y * y) * (1. + x * x + y * y));
                }
        );
    }

    PuassonEquation solver;
    solver.solve(&config);
//    MPI_Finalize();
    return 0;
}