/* z_shift_scale.cc */

#include <Python.h>
#include <numpy/npy_common.h>

int
mapping(npy_intp *output_coordinates, double *input_coordinates,
           int output_rank, int input_rank, void *user_data)
{
    // Get the parameters from the user data:
    // z_start, n_rows, n_cols, flattened shift array, flattened scale array
    // all doubles
    long offset = 0;

    double z_start = *(double *)(user_data + offset); offset += sizeof(double);
    double z_start_input = *(double *)(user_data + offset); offset += sizeof(double);
    double n_rows = *(double *)(user_data + offset); offset += sizeof(double);
    double n_cols = *(double *)(user_data + offset); offset += sizeof(double);

    long offset_shift = ((long)(
        0 +
        output_coordinates[1] * n_rows + output_coordinates[0]
    )) * sizeof(double);
    double shift = *(double *)(user_data + offset_shift);

    long offset_scale = ((long)(
        n_rows * n_cols +
        output_coordinates[1] * n_rows + output_coordinates[0]
    )) * sizeof(double);
    double scale = *(double *)(user_data + offset_scale);

    input_coordinates[0] = output_coordinates[0];
    input_coordinates[1] = output_coordinates[1];
    
    if (scale == 0.0) {
        input_coordinates[2] = output_coordinates[2];
    } else {
        input_coordinates[2] = (z_start + output_coordinates[2] - shift) / scale - z_start_input;
    }
    
    return 1;
}
