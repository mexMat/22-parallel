#include <iostream>
#include <cmath>
#include <mpi.h>
#include <omp.h>


struct Config {

    MPI_Comm comm;

    MPI_Datatype row_type, column_type;

    int coord_x, coord_y;

    int size_comm_x, size_comm_y;

    int block_size_y, block_size_x;

    int init_point_x, init_point_y;

    int end_point_x, end_point_y;

    int rank_left, rank_right, rank_top, rank_bot;

    int rank;

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

    Config(MPI_Comm &comm,
           int coord_x, int coord_y,
           int size_comm_x, int size_comm_y,
           int rank,
           int size_x, int size_y,
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
        this->comm = comm;
        this->coord_x = coord_x;
        this->coord_y = coord_y;
        this->size_comm_x = size_comm_x;
        this->size_comm_y = size_comm_y;
        this->rank = rank;
        this->size_x = size_x;
        this->size_y = size_y;
        //set vector type
        MPI_Type_vector(1, size_x / size_comm_x, 1,
                        MPI_DOUBLE, &this->row_type);
        MPI_Type_vector(size_y / size_comm_y, 1, size_x / size_comm_x,
                        MPI_DOUBLE, &this->column_type);
        MPI_Type_commit(&this->row_type);
        MPI_Type_commit(&this->column_type);
        rank_left = rank;
        rank_right = rank;
        rank_top = rank;
        rank_bot = rank;
        int dims[2] = {coord_x, coord_y};
        if (coord_x != 0) {
            dims[0] -= 1;
            MPI_Cart_rank(comm, dims, &rank_left);
            dims[0] += 1;
        }
        if (coord_x != size_comm_x - 1) {
            dims[0] += 1;
            MPI_Cart_rank(comm, dims, &rank_right);
            dims[0] -= 1;
        }
        if (coord_y != 0) {
            dims[1] -= 1;
            MPI_Cart_rank(comm, dims, &rank_top);
            dims[1] += 1;
        }
        if (coord_y != size_comm_y - 1) {
            dims[1] += 1;
            MPI_Cart_rank(comm, dims, &rank_bot);
            dims[1] -= 1;
        }
        this->block_size_y = size_y / size_comm_y;
        this->block_size_x = size_x / size_comm_x;
        this->init_point_x = block_size_x * coord_x;
        this->init_point_y = block_size_y * coord_y;
        this->end_point_x = init_point_x + block_size_x;
        this->end_point_y = init_point_y + block_size_y;
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
        this->f_func = f_func;
        this->edge_bot = edge_bot;
        this->edge_top = edge_top;
        this->edge_left = edge_left;
        this->edge_right = edge_right;
        this->q_func = q_func;
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
        for (int i = 0; i < size; ++i)
            this->matrix[i] = 0.0;
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

    static void sub(Matrix &a, Matrix &b, Matrix &result, Config *config) {
        #pragma omp parallel for collapse(2)
        for (int i = config->init_point_y; i < config->end_point_y; ++i) {
            for (int j = config->init_point_x; j < config->end_point_x; ++j) {
                result(i, j, 0) = a(i, j, 0) - b(i, j, 0);
            }
        }
    }

    static void add(Matrix &a, Matrix &b, double alpha, double beta, Matrix &result, Config *config) {
        #pragma omp parallel for collapse(2)
        for (int i = config->init_point_y; i < config->end_point_y; ++i) {
            for (int j = config->init_point_x; j < config->end_point_x; ++j) {
                result(i, j, 0) = alpha * a(i, j, 0) + beta * b(i, j, 0);
            }
        }
    }


    static void mul(Matrix &A, Matrix &b, Matrix &top_vector, Matrix &bot_vector,
                    Matrix &left_vector, Matrix &right_vector, Matrix &result, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        int dims[2] = {config->coord_x, config->coord_y};
        int rank_left = 0, rank_right = 0, rank_top = 0, rank_bot = 0;
        int x_left = 0, x_right = M / config->size_comm_x - 1;
        int y_bot = N / config->size_comm_y - 1, y_top = 0;
        double right_point, left_point, bot_point, top_point;
        const int start_y = config->init_point_y, end_y = config->end_point_y;
        const int start_x = config->init_point_x, end_x = config->end_point_x;
        #pragma omp parallel for collapse(2) private(right_point, left_point, bot_point, top_point)
        for (int i = start_y; i < end_y; ++i)
            for (int j = start_x; j < end_x; ++j) {
                // top points
                if (i == 0) {
                    //left top corner
                    if (j == 0) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j + 1, 0) +
                                A(i, j, 1) * b(i + 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    //right top corner
                    if (j == M - 1) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j - 1, 0) +
                                A(i, j, 1) * b(i + 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    left_point = b(i, j - 1, 0);
                    right_point = b(i, j + 1, 0);
                    if (j == start_x)
                        left_point = left_vector(i, 0, 0);
                    if (j == end_x - 1)
                        right_point = right_vector(i, 0, 0);
                    result(i, j, 0) =
                            A(i, j, 0) * b(i + 1, j, 0) +
                            A(i, j, 1) * left_point +
                            A(i, j, 2) * right_point +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                // bot points
                if (i == N - 1) {
                    //right bot
                    if (j == M - 1) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j - 1, 0) +
                                A(i, j, 1) * b(i - 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    //left bot
                    if (j == 0) {
                        result(i, j, 0) =
                                A(i, j, 0) * b(i, j + 1, 0) +
                                A(i, j, 1) * b(i - 1, j, 0) +
                                A(i, j, 2) * b(i, j, 0);
                        continue;
                    }
                    //edge top
                    left_point = b(i, j - 1, 0);
                    right_point = b(i, j + 1, 0);
                    if (j == start_x)
                        left_point = left_vector(i, 0, 0);
                    if (j == end_x - 1)
                        right_point = right_vector(i, 0, 0);
                    result(i, j, 0) =
                            A(i, j, 0) * b(i - 1, j, 0) +
                            A(i, j, 1) * left_point +
                            A(i, j, 2) * right_point +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                //left points
                if (j == 0) {
                    //edge
                    top_point = b(i - 1, j, 0);
                    bot_point = b(i + 1, j, 0);
                    if (i == start_y)
                        top_point = top_vector(0, j, 0);
                    if (i == end_y - 1)
                        bot_point = bot_vector(0, j, 0);
                    result(i, j, 0) =
                            A(i, j, 0) * b(i, j + 1, 0) +
                            A(i, j, 1) * bot_point +
                            A(i, j, 2) * top_point +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                //right points
                if (j == M - 1) {
                    //edge
                    bot_point = b(i + 1, j, 0);
                    top_point = b(i - 1, j, 0);
                    if (i == start_y)
                        top_point = top_vector(0, j, 0);
                    if (i == end_y - 1)
                        bot_point = bot_vector(0, j, 0);
                    result(i, j, 0) =
                            A(i, j, 0) * b(i, j - 1, 0) +
                            A(i, j, 1) * bot_point +
                            A(i, j, 2) * top_point +
                            A(i, j, 3) * b(i, j, 0);
                    continue;
                }
                top_point = b(i - 1, j, 0);
                bot_point = b(i + 1, j, 0);
                left_point = b(i, j - 1, 0);
                right_point = b(i, j + 1, 0);
                if (j == start_x)
                    left_point = left_vector(i, 0, 0);
                if (j == end_x - 1)
                    right_point = right_vector(i, 0, 0);
                if (i == start_y)
                    top_point = top_vector(0, j, 0);
                if (i == end_y - 1)
                    bot_point = bot_vector(0, j, 0);
                // inner points
                result(i, j, 0) =
                        A(i, j, 0) * left_point +
                        A(i, j, 1) * right_point +
                        A(i, j, 2) * top_point +
                        A(i, j, 3) * bot_point +
                        A(i, j, 4) * b(i, j, 0);
            }
    }


    double getItem(int i, int j, int k) const {
        i = i % y_size;
        j = j % x_size;
        return matrix[i * x_size * k_size + j * k_size + k];
    }

    double &operator()(int i, int j, int k) {
        i = i % y_size;
        j = j % x_size;
        return matrix[i * x_size * k_size + j * k_size + k];
    }

    void print(Config *config) const {
        for (int i = config->init_point_y; i < config->end_point_y; ++i)
            for (int j = config->init_point_x; j < config->end_point_x; ++j) {
                for (int k = 0; k < getSizeK(); ++k)
                    std::cout << getItem(i, j, k) << " ";
                std::cout << "(" << i << ", " << j << ")\n";
            }
    }

    void print_plain() const {
        for (int i = 0; i < size; ++i)
            std::cout << matrix[i] << " ";
        std::cout << std::endl;
    }
};

// N - number of rows
// M - number of columns
class PuassonEquation {

    static void sendrecv(Matrix &b, Matrix &top_vector, Matrix &bot_vector, Matrix &left_vector, Matrix &right_vector,
                         Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        int dims[2] = {config->coord_x, config->coord_y};
        int rank_left = 0, rank_right = 0, rank_top = 0, rank_bot = 0;
        int x_left = 0, x_right = M / config->size_comm_x - 1;
        int y_bot = N / config->size_comm_y - 1, y_top = 0;
        MPI_Request request_right, request_left, request_bot, request_top;
        MPI_Status status_right, status_left, status_bot, status_top;
        rank_left = config->rank_left;
        rank_right = config->rank_right;
        rank_top = config->rank_top;
        rank_bot = config->rank_bot;
        if (config->coord_y == 0 or
            config->coord_y == config->size_comm_y - 1) {
            if (config->coord_y == 0) {
                // 0th row block
                MPI_Isend(&b(y_bot, 0, 0), 1, config->row_type,
                          rank_bot, config->rank, config->comm, &request_bot);
                MPI_Recv(&bot_vector(0, 0, 0), config->size_x / config->size_comm_x, MPI_DOUBLE,
                         rank_bot, rank_bot, config->comm, &status_bot);
            }
            // last row block
            if (config->coord_y == config->size_comm_y - 1) {
                MPI_Isend(&b(y_top, 0, 0), 1, config->row_type,
                          rank_top, config->rank, config->comm, &request_top);
                MPI_Recv(&top_vector(0, 0, 0), config->size_x / config->size_comm_x, MPI_DOUBLE,
                         rank_top, rank_top, config->comm, &status_top);
            }
        } else {
            // send row between first and last
            MPI_Isend(&b(y_bot, 0, 0), 1, config->row_type,
                      rank_bot, config->rank, config->comm, &request_bot);
            MPI_Isend(&b(y_top, 0, 0), 1, config->row_type,
                      rank_top, config->rank, config->comm, &request_top);
            // recv rows between first and last
            MPI_Recv(&top_vector(0, 0, 0), config->size_x / config->size_comm_x, MPI_DOUBLE,
                     rank_top, rank_top, config->comm, &status_top);
            MPI_Recv(&bot_vector(0, 0, 0), config->size_x / config->size_comm_x, MPI_DOUBLE,
                     rank_bot, rank_bot, config->comm, &status_bot);
        }

        // sendrecv vec to/from right cross x dimension
        if (config->coord_x == 0 or
            config->coord_x == config->size_comm_x - 1) {

            // 0th column block
            if (config->coord_x == 0) {
                MPI_Isend(&b(0, x_right, 0), 1, config->column_type,
                          rank_right, config->rank, config->comm, &request_right);
                MPI_Recv(&right_vector(0, 0, 0), config->size_y / config->size_comm_y, MPI_DOUBLE,
                         rank_right, rank_right, config->comm, &status_right);
            }
            // last column block
            if (config->coord_x == config->size_comm_x - 1) {
                MPI_Isend(&b(0, x_left, 0), 1, config->column_type,
                          rank_left, config->rank, config->comm, &request_left);
                MPI_Recv(&left_vector(0, 0, 0), config->size_y / config->size_comm_y, MPI_DOUBLE,
                         rank_left, rank_left, config->comm, &status_left);
            }
        } else {
            // send columns between first and last
            MPI_Isend(&b(0, x_right, 0), 1, config->column_type,
                      rank_right, config->rank, config->comm, &request_right);
            MPI_Isend(&b(0, x_left, 0), 1, config->column_type,
                      rank_left, config->rank, config->comm, &request_left);
            // recv columns between first and last
            MPI_Recv(&left_vector(0, 0, 0), config->size_y / config->size_comm_y, MPI_DOUBLE,
                     rank_left, rank_left, config->comm, &status_left);
            MPI_Recv(&right_vector(0, 0, 0), config->size_y / config->size_comm_y, MPI_DOUBLE,
                     rank_right, rank_right, config->comm, &status_right);
        }
    }

    static double dot(Matrix &a, Matrix &b, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        double h1 = config->h_1;
        double h2 = config->h_2;
        double p1, p2;
        double sum_e2 = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:sum_e2)
        for (int i = config->init_point_y; i < config->end_point_y; ++i)
            for (int j = config->init_point_x; j < config->end_point_x; ++j) {
                if ((i == 0) or (i == N - 1))
                    p2 = 0.5;
                else
                    p2 = 1.;
                if ((j == 0) or (j == M - 1))
                    p1 = 0.5;
                else
                    p1 = 1.;
                sum_e2 += h1 * h2 * p1 * p2 * a(i, j, 0) * b(i, j, 0);
            }
        return sum_e2;
    }

    static double norm_c(Matrix &a, Config *config) {
        double max_val = a(0, 0, 0);
        #pragma omp parallel for collapse(2) reduction(max:max_val)
        for (int i = config->init_point_y; i < config->end_point_y; ++i)
            for (int j = config->init_point_x; j < config->end_point_x; ++j) {
                if (fabs(a(i, j, 0)) > max_val)
                    max_val = fabs(a(i, j, 0));
            }
        return max_val;
    }

    static double norm(Matrix &a, Config *config) {
        return dot(a, a, config);
    }

    static void filling(Matrix &matrix, Matrix &f_vector, Config *config) {
        int N = config->size_y;
        int M = config->size_x;
        double h1 = config->h_1;
        double h2 = config->h_2;
        double h1_2 = h1 * h1, h2_2 = h2 * h2;
        double x_left = config->x_left, x_right = config->x_right;
        double y_bot = config->y_bot, y_top = config->y_top;
        double xi, yj;
        for (int i = config->init_point_y; i < config->end_point_y; ++i)
            for (int j = config->init_point_x; j < config->end_point_x; ++j) {
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

    static void optimize(Matrix &matrix, Matrix &f_vector, Matrix &w_vector,
                         Matrix &r_vector, Matrix &Ar_vector, Matrix &true_solution,
                         Matrix &top_vector, Matrix &bot_vector, Matrix &left_vector,
                         Matrix &right_vector, Config *config) {

        double tau_ar_proc[2] = {0., 0.}, tau_ar[2] = {0., 0.}, tau_k;
        double eps_proc = 0., eps = 10.;
        int iter = 0;
        while (eps > 1e-06) {
            PuassonEquation::sendrecv(w_vector, top_vector, bot_vector, left_vector, right_vector, config);
            Matrix::mul(matrix, w_vector, top_vector, bot_vector,
                        left_vector, right_vector, Ar_vector, config);
            Matrix::sub(Ar_vector, f_vector, r_vector, config);
            PuassonEquation::sendrecv(r_vector, top_vector, bot_vector, left_vector, right_vector, config);
            Matrix::mul(matrix, r_vector, top_vector, bot_vector,
                        left_vector, right_vector, Ar_vector, config);
            tau_ar_proc[0] = dot(Ar_vector, r_vector,  config);
            tau_ar_proc[1] = dot(Ar_vector, Ar_vector,  config);
            MPI_Allreduce(&tau_ar_proc, &tau_ar, 2, MPI_DOUBLE, MPI_SUM, config->comm);
            tau_k = tau_ar[0]/tau_ar[1];
            Matrix::add(w_vector, r_vector, 1.0, -tau_k, w_vector, config);
            //e2 norm
            eps_proc = norm(r_vector,  config);
            MPI_Allreduce(&eps_proc, &eps, 1, MPI_DOUBLE, MPI_SUM, config->comm);
            eps = sqrt(eps);
            ++iter;
        }
        if (config->rank == 0) {
            std::cout<<"iter: "<<iter<<std::endl;
//            w_vector.print(config);
        }
    }


    static void set_true_solution(Matrix &a, Config *config) {
        double h1 = config->h_1;
        double h2 = config->h_2;
        double x_left = config->x_left;
        double y_bot = config->y_bot;
        double xi, yj;
        for (int i = config->init_point_y; i < config->end_point_y; ++i)
            for (int j = config->init_point_x; j < config->end_point_x; ++j) {
                xi = x_left + j * h1;
                yj = y_bot + i * h2;
                a(i, j, 0) = config->u_func(xi, yj);
            }
    }

public:

    static void solve(Config *config) {
        Matrix matrix(config->size_x / config->size_comm_x,
                      config->size_y / config->size_comm_y, 5);
        Matrix f_vector(config->size_x / config->size_comm_x,
                        config->size_y / config->size_comm_y, 1);
        Matrix w_vector(config->size_x / config->size_comm_x,
                        config->size_y / config->size_comm_y, 1);
        Matrix r_vector(config->size_x / config->size_comm_x,
                        config->size_y / config->size_comm_y, 1);
        Matrix Ar_vector(config->size_x / config->size_comm_x,
                         config->size_y / config->size_comm_y, 1);
        Matrix bot_vector(config->size_x / config->size_comm_x, 1, 1);
        Matrix top_vector(config->size_x / config->size_comm_x, 1, 1);
        Matrix left_vector(1, config->size_y / config->size_comm_y, 1);
        Matrix right_vector(1, config->size_y / config->size_comm_y, 1);
        Matrix true_solution(config->size_x / config->size_comm_x,
                             config->size_y / config->size_comm_y, 1);
        set_true_solution(true_solution, config);
        double start_time = MPI_Wtime();
        filling(matrix, f_vector, config);
        optimize(matrix, f_vector, w_vector, r_vector, Ar_vector, true_solution,
                 top_vector, bot_vector, left_vector, right_vector, config);
        double end_time = MPI_Wtime();
        double result_time, time = end_time - start_time;
        MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        Matrix::sub(true_solution, w_vector, r_vector, config);
        double error_e2_proc = 0., error_e2 = 0.;
        error_e2_proc = norm(r_vector,  config);
        double error_max_proc = 0., error_max = 0.;
        error_max_proc = norm_c(r_vector,  config);
        MPI_Reduce(&error_e2_proc, &error_e2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&error_max_proc, &error_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (config->rank == 0) {
            std::cout << "result time: " << result_time << std::endl;
            std::cout << "abs norm e2: " << sqrt(error_e2) << std::endl;
            std::cout << "abs norm max: " << error_max << std::endl;
        }
    }

};


int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims_size[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 0};

    MPI_Comm comm;
    MPI_Dims_create(size, 2, dims_size);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims_size, periods,
                    1, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);

    //my config
    Config config(
            comm,
            coords[0], coords[1],
            dims_size[0], dims_size[1],
            rank,
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
    PuassonEquation::solve(&config);
    MPI_Finalize();
    return 0;
}
