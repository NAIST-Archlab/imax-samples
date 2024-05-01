K=16
D=16
N=16777216
init_K=32

nvprof -o logfile.nvvp ./a.out ../evaluation_VBGMM/data/txt/data_${K}_${D}_${N}.txt ${init_K} ../evaluation_VBGMM/rand/txt/data_${init_K}_${D}_1.txt 


