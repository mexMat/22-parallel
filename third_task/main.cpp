#include <iostream>
#include <array>
#include <cmath>

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
           double (*u_func)(double, double), double (*k_func)(double, double),
           double (*q_func)(double, double), double (*f_func)(double, double),
           double (*edge_top)(double, double), double (*edge_bot)(double, double),
           double (*edge_left)(double, double), double (*edge_right)(double, double)) {
        this->size_x = size_x;
        this->size_y = size_y;
        this->h_1 = (double) (x_right - x_left) / (double) size_x;
        this->h_2 = (double) (y_top - y_bot) / (double) size_y;
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
        for (int i = 0; i < this->x_size; ++i)
            for (int j = 0; j < this->y_size; ++j) {
                for (int k = 0; k < this->k_size; ++k)
                    std::cout << this->getItem(i, j, k) << " ";
                std::cout << "(" << i << ", " << j << ")\n";
            }
    }
};

// N - number of rows
// M - number of columns
class PuassonEquation {
    static Matrix filling(Matrix &matrix, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        double h1 = config->h_1;
        double h2 = config->h_2;
        double h1_2 = h1 * h1, h2_2 = h2 * h2;
        double x_left = config->x_left, x_right = config->x_right;
        double y_bot = config->y_bot, y_top = config->y_top;
        double xi = 0, yj = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                xi = x_left + j * h1;
                yj = y_bot + i * h2;
                // bottom points
                if (i == 0) {
                    //left bottom
                    if (j == 0) {
                        matrix(i, j, 0) = -(2. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                        matrix(i, j, 1) = -(2. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_L / h1 +
                                2. * config->alpha_B / h2;
                        continue;
                    }
                    //right bottom
                    if (j == M - 1) {
                        matrix(i, j, 0) = -(2. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                        matrix(i, j, 1) = -(2. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_R / h1 +
                                2. * config->alpha_B / h2;
                        continue;
                    }
                    //edge
                    matrix(i, j, 0) = -(2. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 1) = -(1. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 2) = -(1. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_B / h2;
                    continue;
                }
                // top points
                if (i == N - 1) {
                    //right top
                    if (j == M - 1) {
                        matrix(i, j, 0) = -(2. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                        matrix(i, j, 1) = -(2. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_R / h1 +
                                2. * config->alpha_T / h2;
                        continue;
                    }
                    //left top
                    if (j == 0) {
                        matrix(i, j, 0) = -(2. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                        matrix(i, j, 1) = -(2. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                        matrix(i, j, 2) =
                                -matrix(i, j, 0) -
                                matrix(i, j, 1) +
                                config->q_func(xi, yj) +
                                2. * config->alpha_L / h1 +
                                2. * config->alpha_T / h2;
                        continue;
                    }
                    //edge
                    matrix(i, j, 0) = -(2. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 1) = -(1. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 2) = -(1. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_T / h2;
                    continue;
                }

                //left points
                if (j == 0) {
                    //edge
                    matrix(i, j, 0) = -(2. / h1_2) * config->k_func(xi + 0.5 * h1, yj);
                    matrix(i, j, 1) = -(1. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 2) = -(1. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_L / h1;
                    continue;
                }

                //right points
                if (j == M - 1) {
                    //edge
                    matrix(i, j, 0) = -(2. / h1_2) * config->k_func(xi - 0.5 * h1, yj);
                    matrix(i, j, 1) = -(1. / h2_2) * config->k_func(xi, yj + 0.5 * h2);
                    matrix(i, j, 2) = -(1. / h2_2) * config->k_func(xi, yj - 0.5 * h2);
                    matrix(i, j, 3) =
                            -matrix(i, j, 0) -
                            matrix(i, j, 1) -
                            matrix(i, j, 2) +
                            config->q_func(xi, yj) +
                            2. * config->alpha_R / h1;
                    continue;
                }

                //inner points
                matrix(i, j, 0) = 1. / (h1_2) * config->k_func(xi - 0.5 * h1, yj);
                matrix(i, j, 1) = 1. / (h1_2) * config->k_func(xi + 0.5 * h1, yj);
                matrix(i, j, 2) = 1. / (h2_2) * config->k_func(xi, yj - 0.5 * h2);
                matrix(i, j, 3) = 1. / (h2_2) * config->k_func(xi, yj + 0.5 * h2);
                matrix(i, j, 4) =
                        -(matrix(i, j, 0) + matrix(i, j, 1)) -
                        (matrix(i, j, 2) + matrix(i, j, 3));
            }
    }


public:
    static Matrix solve(Config *config) {
        int NM = config->size_x * config->size_y;
        Matrix matrix(config->size_y, config->size_x, 5);
        Matrix vector(config->size_y, config->size_x, 1);
        filling(matrix, config);
        matrix.print();
    }
};

int main(int argc, char *argv[]) {
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
                return x * x * (x + y + 4.) / (4. * pow(x * y + 4., 3. / 2.)) +
                       y * y * (x + y + 4.) / (4. * pow(x * y + 4, 3. / 2.)) -
                       x / (2. * pow(x * y + 4., 0.5)) -
                       y / (2. * pow(x * y + 4., 0.5)) +
                       (x + y) * pow(x * y + 4., 0.5);
            },
            [](double x, double y) -> double { return x + y; },
            [](double x, double y) -> double { return x + y; },
            [](double x, double y) -> double { return x + y; },
            [](double x, double y) -> double { return x + y; }
    );
    PuassonEquation solver;
    solver.solve(&config);
    return 0;
}