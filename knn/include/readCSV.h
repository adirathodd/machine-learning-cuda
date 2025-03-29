#ifndef READCSV_H
#define READCSV_H

#ifdef __cplusplus
extern "C" {
#endif

float *loadCSV(const char *filename);

extern int numRows;
extern int numCols;

#ifdef __cplusplus
}
#endif

#endif /* READCSV_H */

