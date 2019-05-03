#!/bin/bash

date > out.txt

for kmeans in {16,32,64,128} ; do
    for step_size in {0,2,3,4,7,15} ; do
        for knn in {1,2,4,16} ; do
            echo $kmeans $step_size $knn >> out.txt
            python3 k_nearest_neighbors.py dataset/ $kmeans $step_size $knn >> out.txt
        done
    done
done

# python3 k_means_clustering.py dataset/ 128 0
# echo "128 0 1" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 0 1 >> out.txt
# echo "128 0 2" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 0 2 >> out.txt
# echo "128 0 4" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 0 4 >> out.txt
# echo "128 0 16" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 0 16 >> out.txt

# echo "" >> out.txt

# python3 k_means_clustering.py dataset/ 128 2
# echo "128 2 1" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 2 1 >> out.txt
# echo "128 2 2" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 2 2 >> out.txt
# echo "128 2 4" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 2 4 >> out.txt
# echo "128 2 16" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 2 16 >> out.txt

# echo "" >> out.txt

# python3 k_means_clustering.py dataset/ 128 3
# echo "128 3 1" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 3 1 >> out.txt
# echo "128 3 2" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 3 2 >> out.txt
# echo "128 3 4" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 3 4 >> out.txt
# echo "128 3 16" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 3 16 >> out.txt

# echo "" >> out.txt

# python3 k_means_clustering.py dataset/ 128 4
# echo "128 4 1" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 4 1 >> out.txt
# echo "128 4 2" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 4 2 >> out.txt
# echo "128 4 4" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 4 4 >> out.txt
# echo "128 4 16" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 4 16 >> out.txt

# echo "" >> out.txt

# python3 k_means_clustering.py dataset/ 128 7
# echo "128 7 1" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 7 1 >> out.txt
# echo "128 7 2" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 7 2 >> out.txt
# echo "128 7 4" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 7 4 >> out.txt
# echo "128 7 16" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 7 16 >> out.txt

# echo "" >> out.txt

# python3 k_means_clustering.py dataset/ 128 15
# echo "128 15 1" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 15 1 >> out.txt
# echo "128 15 2" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 15 2 >> out.txt
# echo "128 15 4" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 15 4 >> out.txt
# echo "128 15 16" >> out.txt
# python3 k_nearest_neighbors.py dataset/ 128 15 16 >> out.txt

date >> out.txt