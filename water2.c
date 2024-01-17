#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

#define LENGTH 80
const int maxnum = 100000;
double r[maxnum][3][3], rcutsq = 1.44, L;

double sqr(double a) { return a * a; }

double energy12(int i1, int i2)
{
    int m, n, xyz;
    double shift[3], dr[3], mn[3], r6, distsq, dist, ene = 0;
    const double sig = 0.3166, eps = 0.65, eps0 = 8.85e-12, e = 1.602e-19, Na = 6.022e23, q[3] = {-0.8476, 0.4238, 0.4238};
    double elst, sig6;
    elst = e * e / (4 * 3.141593 * eps0 * 1e-9) * Na / 1e3, sig6 = pow(sig, 6);

    // periodic boundary conditions
    for (xyz = 0; xyz <= 2; xyz++)
    {
        dr[xyz] = r[i1][0][xyz] - r[i2][0][xyz];
        shift[xyz] = -L * floor(dr[xyz] / L + 0.5); // round dr[xyz]/L to nearest integer
        dr[xyz] = dr[xyz] + shift[xyz];
    }

    distsq = sqr(dr[0]) + sqr(dr[1]) + sqr(dr[2]);

    if (distsq < rcutsq && distsq > 1e-10)
    { // calculate energy if within cutoff and distance is not too small
        r6 = sig6 / pow(distsq, 3);
        ene = 4 * eps * r6 * (r6 - 1.); // LJ energy

        for (m = 0; m <= 2; m++)
        {
            for (n = 0; n <= 2; n++)
            {
                for (xyz = 0; xyz <= 2; xyz++)
                    mn[xyz] = r[i1][m][xyz] - r[i2][n][xyz] + shift[xyz];

                dist = sqrt(sqr(mn[0]) + sqr(mn[1]) + sqr(mn[2]));

                if (dist > 1e-10)
                { // avoid division by zero
                    ene += elst * q[m] * q[n] / dist;
                }
            }
        }
    }
    return ene;
}

int main(int argc, char *argv[])
{
    int i, j, natoms, nmol, rank, size;
    double energy = 0, dtime;
    clock_t cputime;
    struct timeval start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE *fp;
    char line[LENGTH], nothing[LENGTH], name[20];

    if (rank == 0)
    {
        printf("Program to calculate energy of water\n");
        printf("Input NAME of configuration file : ");
        scanf("%s", name);       // reading of filename from keyboard
        fp = fopen(name, "r");   // opening of file and beginning of reading from HDD
        fgets(line, LENGTH, fp); // skip first line
        fgets(line, LENGTH, fp);
        sscanf(line, "%i", &natoms);

        nmol = natoms / 3;
        printf("\nNumber of molecules %i\n", nmol);

        for (i = 0; i < nmol; i++)
        {
            for (j = 0; j <= 2; j++)
            {
                fgets(line, LENGTH, fp);
                sscanf(line, "%s %s %s %lf %lf %lf", nothing, nothing, nothing, &r[i][j][0], &r[i][j][1], &r[i][j][2]);
            }
        }
        printf("first line %lf %lf %lf\n", r[0][0][0], r[0][0][1], r[0][0][2]);
        fscanf(fp, "%lf", &L); // read box size
        printf("Box size %lf\n", L);

        fclose(fp);
    }

    MPI_Bcast(&nmol, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int chunk_size = nmol / size;
    int start_idx = rank * chunk_size;
    int end_idx = (rank == size - 1) ? nmol : start_idx + chunk_size;

    double local_energy = 0;

    cputime = clock();
    gettimeofday(&start_time, NULL);

    for (i = start_idx; i < end_idx - 1; i++)
    {
        for (j = i + 1; j < nmol; j++)
        {
            local_energy += energy12(i, j);
        }
    }

    cputime = clock() - cputime;
    gettimeofday(&end_time, NULL);
    dtime = ((end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6);

    MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Total energy %lf \n ", energy);
        printf("Energy per molecule %lf \n", energy / nmol);
        printf("Elapsed wall time: %f\n", dtime);
        printf("Elapsed CPU  time: %f\n", (float)cputime / CLOCKS_PER_SEC);
    }

    MPI_Finalize();

    return 0;
}
