// g++ --std=c++11 rna_mp.cpp -o rna_mp -fopenmp
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <omp.h> // Include OpenMP header

// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total0(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    
                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !
                    dp[k][j] += dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
        }
        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}


// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total1(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                //#pragma omp parallel for schedule(dynamic) // it's faster not having this parallel; but why?

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    long long a = dp[k][i-2] * dp[i][j-1];

                    #pragma omp atomic update
                    dp[k][j] += a; //single-element write lock
                }
            }
        }
        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total2(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.


        // ALLOCATING 2D mem space costs O(n^2) time, for every i
        std::vector<std::vector<long long>> dp_tmp(n + 1, std::vector<long long>(n + 1, 0));

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    dp_tmp[k][i] = dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
        }

        #pragma omp parallel for schedule(dynamic)        
        // parallel sum
        for (int k = 1; k <= j; ++k) {            
            for (int i=k+1; i<=j;i++)
                dp[k][j] += dp_tmp[k][i];
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total3(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        // ALLOCATING 2D mem space costs O(n^2) time, for every i
        std::vector<std::vector<long long>> dp_tmp(n + 1, std::vector<long long>(n + 1, 0));

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    dp_tmp[k][i] = dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int k = 1; k <= j; ++k) {
            long long local = dp[k][j];        

            #pragma omp simd reduction(+:local)
            for (int i = k + 1; i <= j; ++i) {
                local += dp_tmp[k][i];
            }

            dp[k][j] = local;
        }

        

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total4(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        // ALLOCATING 2D mem space costs O(n^2) time, for every i
        std::vector<std::vector<long long>> dp_tmp(n + 1, std::vector<long long>(n + 1, 0));

        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                #pragma omp parallel for schedule(static)
                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    dp_tmp[k][i] = dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int k = 1; k <= j; ++k) {
            long long local = dp[k][j];        

            #pragma omp parallel for reduction(+:local)
            for (int i = k + 1; i <= j; ++i) {
                local += dp_tmp[k][i];
            }

            dp[k][j] = local;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total5(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        // ALLOCATING 2D mem space costs O(n^2) time, for every i
        std::vector<long long> dp_tmp((n + 1) * (n + 1), 0); // flattened 2D

        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i) {
            dp[i][j] = dp[i][j - 1];
        }

        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i) {
            std::string pair = {s[i - 1], s[j]};
            if (allowed.count(pair)) {
                for (int k = 1; k <= i - 1; ++k) {
                    dp_tmp[k * (n + 1) + i] = dp[k][i - 2] * dp[i][j - 1];
                }
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int k = 1; k <= j; ++k) {
            long long local = dp[k][j];

            #pragma omp simd reduction(+:local)
            for (int i = k + 1; i <= j; ++i) {
                local += dp_tmp[k * (n + 1) + i];
            }

            dp[k][j] = local;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

// Function to count RNA secondary structures using the KT formulation, parallelized over i for each span.
long long total6(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        #pragma omp parallel
        {
            std::vector<std::tuple<int, int, long long>> local_updates;

            #pragma omp for schedule(dynamic) nowait
            for (int i = 1; i <= j; ++i) {
                std::string pair = {s[i - 1], s[j]};
                if (allowed.count(pair)) {
                    for (int k = 1; k <= i - 1; ++k) {
                        long long a = dp[k][i - 2] * dp[i][j - 1];
                        local_updates.emplace_back(k, j, a);
                    }
                }
            }

            #pragma omp critical
            {
                for (const auto& tup : local_updates) {
                    int k, j_idx;
                    long long val;
                    std::tie(k, j_idx, val) = tup;
                    dp[k][j_idx] += val;
                }
            }
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

long long total7(std::string s, bool verbose) {
    int n = s.size();
    s = " " + s; // 1-based indexing

    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));
    for (int j = 1; j <= n; ++j)
        dp[j][j - 1] = 1;

    for (int j = 1; j <= n; j++) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i)
            dp[i][j] = dp[i][j - 1];

        int num_threads = omp_get_max_threads();
        std::vector<std::unordered_map<int, long long>> thread_maps(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_map = thread_maps[tid];

            #pragma omp for schedule(dynamic)
            for (int i = 1; i <= j; ++i) {
                std::string pair = {s[i - 1], s[j]};
                if (allowed.count(pair)) {
                    for (int k = 1; k <= i - 1; ++k) {
                        long long a = dp[k][i - 2] * dp[i][j - 1];
                        local_map[k] += a;
                    }
                }
            }
        }

        for (const auto& map : thread_maps) {
            for (const auto& kv : map) {
                int k = kv.first;
                long long val = kv.second;
                dp[k][j] += val;
            }            
        }


        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i)
                std::cout << i << "-" << j << ":" << dp[i][j] << " ";
            std::cout << std::endl;
        }
    }

    return dp[1][n];
}

long long total8(std::string s, bool verbose) {
    int n = s.size();
    s = " " + s; // 1-based indexing

    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));
    for (int j = 1; j <= n; ++j)
        dp[j][j - 1] = 1;

    for (int j = 1; j <= n; j++) {
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i)
            dp[i][j] = dp[i][j - 1];

        int num_threads = omp_get_max_threads();
        std::vector<std::unordered_map<int, long long>> thread_maps(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_map = thread_maps[tid];

            #pragma omp for schedule(dynamic) nowait
            for (int i = 1; i <= j; ++i) {
                std::string pair = {s[i - 1], s[j]};
                if (allowed.count(pair)) {
                    for (int k = 1; k <= i - 1; ++k) {
                        long long a = dp[k][i - 2] * dp[i][j - 1];
                        local_map[k] += a;
                    }
                }
            }
        }

        // real    2m46.380s
        // user    22m1.209s
        // sys     0m0.413s
        // for (const auto& map : thread_maps) {
        //     for (const auto& kv : map) {
        //         int k = kv.first;
        //         long long val = kv.second;
        //         dp[k][j] += val;
        //     }            
        // }

        // real    2m47.116s
        // user    22m6.725s
        // sys     0m0.413s
        // #pragma omp parallel for
        // for (int k = 1; k <= j; ++k) {
        //     for (int t = 0; t < num_threads; ++t) {
        //         if (thread_maps[t].count(k))
        //             dp[k][j] += thread_maps[t][k];
        //     }
        // }

        // real    3m11.626s
        // user    25m16.337s
        // sys     0m6.782s
        // for (int k = 1; k <= j; ++k) {
        //     long long local = 0;

        //     #pragma omp parallel for reduction(+:local)
        //     for (int i = 0; i < num_threads; ++i) {
        //         if (thread_maps[i].count(k))
        //             local += thread_maps[i][k];
        //     }
        //     dp[k][j] += local;
        // }
        
        // real    2m47.263s
        // user    22m1.518s
        // sys     0m7.419s
        #pragma omp parallel for schedule(dynamic)
        for (int k = 1; k <= j; ++k) {
            long long local = 0;

            #pragma omp parallel for reduction(+:local)
            for (int i = 0; i < num_threads; ++i) {
                if (thread_maps[i].count(k))
                    local += thread_maps[i][k];
            }
            dp[k][j] += local;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i)
                std::cout << i << "-" << j << ":" << dp[i][j] << " ";
            std::cout << std::endl;
        }
    }

    return dp[1][n];
}

long long total9(std::string s, bool verbose) {
    int n = s.size();
    s = " " + s; // now 1-based

    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    for (int j = 1; j <= n; ++j)
        dp[j][j - 1] = 1;

    for (int j = 1; j <= n; j++) {
        // Case 1: j is unpaired
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i)
            dp[i][j] = dp[i][j - 1];

        // Prepare per-thread vector of (k, a) updates
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::pair<int, long long>>> thread_updates(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& updates = thread_updates[tid];

            #pragma omp for schedule(dynamic) nowait
            for (int i = 1; i <= j; ++i) {
                std::string pair = {s[i - 1], s[j]};
                if (allowed.count(pair)) {
                    for (int k = 1; k <= i - 1; ++k) {
                        long long a = dp[k][i - 2] * dp[i][j - 1];
                        updates.emplace_back(k, a);
                    }
                }
            }
        }


        // real    4m50.155s
        // user    22m33.305s
        // sys     0m4.761s
        for (const auto& pair : thread_updates) {
            for (const auto& kv : pair) {
                int k = kv.first;
                long long val = kv.second;
                dp[k][j] += val;
            }            
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i)
                std::cout << i << "-" << j << ":" << dp[i][j] << " ";
            std::cout << std::endl;
        }
    }

    return dp[1][n];
}

long long total10(std::string s, bool verbose) {
    int n = s.size();
    s = " " + s; // now 1-based

    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    for (int j = 1; j <= n; ++j)
        dp[j][j - 1] = 1;

    int num_threads = omp_get_max_threads();
    for (int j = 1; j <= n; ++j) {
        // Case 1: j is unpaired
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i)
            dp[i][j] = dp[i][j - 1];

        // Per-thread, per-k list of values
        std::vector<std::vector<std::vector<long long>>> thread_updates(num_threads, std::vector<std::vector<long long>>(n + 1));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& updates = thread_updates[tid];

            #pragma omp for schedule(dynamic) nowait
            for (int i = 1; i <= j; ++i) {
                std::string pair = {s[i - 1], s[j]};
                if (allowed.count(pair)) {
                    for (int k = 1; k <= i - 1; ++k) {
                        long long a = dp[k][i - 2] * dp[i][j - 1];
                        updates[k].push_back(a);
                    }
                }
            }
        }

        // real    2m36.347s
        // user    17m57.103s
        // sys     0m18.885s
        #pragma omp parallel for schedule(dynamic)
        for (int k = 1; k <= j; ++k) {
            long long total = 0;
            for (int t = 0; t < num_threads; ++t) {
                for (long long a : thread_updates[t][k]) {
                    total += a;
                }
            }
            dp[k][j] += total;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i)
                std::cout << i << "-" << j << ":" << dp[i][j] << " ";
            std::cout << std::endl;
        }
    }

    return dp[1][n];
}

long long total11(std::string s, bool verbose) {
    int n = s.size();
    s = " " + s; // now 1-based

    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    for (int j = 1; j <= n; ++j)
        dp[j][j - 1] = 1;

    int num_threads = omp_get_max_threads();

    for (int j = 1; j <= n; ++j) {
        // Case 1: j is unpaired
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i)
            dp[i][j] = dp[i][j - 1];

        // thread_updates[t][k] = total contribution to dp[k][j] by thread t
        std::vector<std::vector<long long>> thread_updates(num_threads, std::vector<long long>(n + 1, 0));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local = thread_updates[tid];

            #pragma omp for schedule(dynamic) nowait
            for (int i = 1; i <= j; ++i) {
                std::string pair = {s[i - 1], s[j]};
                if (allowed.count(pair)) {
                    for (int k = 1; k <= i - 1; ++k) {
                        long long a = dp[k][i - 2] * dp[i][j - 1];
                        local[k] += a;
                    }
                }
            }
        }

        // real    1m8.373s
        // user    9m3.826s
        // sys     0m0.560s
        // #pragma omp parallel for schedule(dynamic)
        // for (int k = 1; k <= j; ++k) {
        //     long long total = 0;
        //     for (int t = 0; t < num_threads; ++t) {
        //         total += thread_updates[t][k];
        //     }
        //     dp[k][j] += total;
        // }

        // real    1m9.349s
        // user    9m4.388s
        // sys     0m8.118s
        #pragma omp parallel for schedule(dynamic)
        for (int k = 1; k <= j; ++k) {
            long long total = 0;

            #pragma omp parallel for reduction(+:total)
            for (int t = 0; t < num_threads; ++t) {
                total += thread_updates[t][k];
            }
            dp[k][j] += total;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i)
                std::cout << i << "-" << j << ":" << dp[i][j] << " ";
            std::cout << std::endl;
        }
    }

    return dp[1][n];
}


// throw to 2D array (vector), then collect
long long total12(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // this line has to be outside; otherwise mem alloc is too slow  
    std::vector<std::vector<long long>> dp_tmp(n + 1, std::vector<long long>(n + 1, 0));

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        //dp_tmp.clear(); // runtime error
        // #pragma omp parallel for schedule(dynamic), collapse(2)
        // for (int i = 1; i <= j; ++i)
        //     for (int k=1; k <= i-1; ++k)
        //         dp_tmp[k][i] = 0; 


        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        
        #pragma omp parallel for schedule(dynamic)

        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                //#pragma omp parallel for schedule(dynamic) // doesn't matter whether this line or not

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    dp_tmp[k][i] = dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
            else { // can be faster!
                #pragma omp parallel for schedule(dynamic)                
                for (int k=1; k<=i-1; ++k)
                    dp_tmp[k][i] = 0;
            }

        }

        #pragma omp parallel for schedule(dynamic) // important
        // parallel sum
        for (int k = 1; k <= j; ++k) {       
            long long a = 0;
            #pragma omp parallel for shared(k,j),reduction(+:a) // seems no use
            for (int i=k+1; i<=j;i++) 
                //if (allowed.count({s[i-1], s[j]})) // too slow
                a += dp_tmp[k][i];
            dp[k][j] += a;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}


// try implement array flatenning on this and iflag to mark an i to be unpaired for a j
// throw to 2D array (vector), then collect
long long total13(std::string s, bool verbose) {
    int n = s.size();

    s = " " + s; // now 1-based

    // Allowed pairs as two-character strings.
    std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

    // Create a 2D DP table with dimensions (n+1) x (n+1), initialized to 0.
    std::vector<std::vector<long long>> dp(n + 1, std::vector<long long>(n + 1, 0));

    // follow LP pseudocode:
    // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)    
    for (int j = 1; j <= n; ++j) 
        dp[j][j-1] = 1;

    // this line has to be outside; otherwise mem alloc is too slow  
    std::vector<std::vector<long long>> dp_tmp(n + 1, std::vector<long long>(n + 1, 0));

    // left to right
    for (int j = 1; j <= n; j++) {
        // Parallelize the loop over i for the current span.

        //dp_tmp.clear(); // runtime error
        // #pragma omp parallel for schedule(dynamic), collapse(2)
        // for (int i = 1; i <= j; ++i)
        //     for (int k=1; k <= i-1; ++k)
        //         dp_tmp[k][i] = 0; 


        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i) {            
            // Case 1: skip (j is unpaired)
            dp[i][j] = dp[i][j-1];
        }

        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i <= j; ++i) {            

            // Case 2: {i-1} pair with j? 
            std::string pair = {s[i-1], s[j]};
            if (allowed.count(pair)) {    

                //#pragma omp parallel for schedule(dynamic) // doesn't matter whether this line or not

                for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

                    dp_tmp[k][i] = dp[k][i-2] * dp[i][j-1]; //single-element write lock
                }
            }
            else { // can be faster!
                #pragma omp parallel for schedule(dynamic)                
                for (int k=1; k<=i-1; ++k)
                    dp_tmp[k][i] = 0;
            }

        }

        #pragma omp parallel for schedule(dynamic) // important
        // parallel sum
        for (int k = 1; k <= j; ++k) {       
            long long a = 0;
            #pragma omp parallel for shared(k,j),reduction(+:a) // seems no use
            for (int i=k+1; i<=j;i++) 
                //if (allowed.count({s[i-1], s[j]})) // too slow
                a += dp_tmp[k][i];
            dp[k][j] += a;
        }

        if (verbose) {
            std::cout << "j=" << j << std::endl;
            for (int i = 1; i <= j; ++i) 
                std::cout << i<<"-"<<j<<":"<< dp[i][j] << " ";
            std::cout << std::endl;
        }
    }
    return dp[1][n];
}

long long total14(std::string s, bool verbose) {
    int n = s.size();
    s = " " + s;             // 1-based
    std::unordered_set<std::string> allowed = {"AU","UA","CG","GC","GU","UG"};
    std::vector<std::vector<long long>> dp(n+1, std::vector<long long>(n+1,0));

    for(int j=1; j<=n; ++j)
        dp[j][j-1] = 1;

    for(int j=1; j<=n; ++j) {
        // case where j is unpaired and build list of valid i's
        std::vector<int> valid_i;
        valid_i.reserve(j);
        for(int i=1; i<=j; ++i) {
            dp[i][j] = dp[i][j-1];                  
            if (allowed.count({s[i-1], s[j]}))
                valid_i.push_back(i);
        }


        // real    0m43.754s
        // user    5m48.219s
        // sys     0m0.110s
        //for each k, sum over only the valid i's
        // #pragma omp parallel for schedule(dynamic)
        // for(int k=1; k<=j; ++k) {
        //     long long sum = 0;
        //     for(int idx=0; idx<valid_i.size(); ++idx) {
        //         int i = valid_i[idx];
        //         if (k < i) {                            // important case
        //             // dp_tmp[k][i] = dp[k][i-2] * dp[i][j-1];
        //             sum += dp[k][i-2] * dp[i][j-1];
        //         }
        //     }
        //     dp[k][j] += sum;
        // }

        // real    0m45.758s
        // user    6m3.706s
        // sys     0m0.121s
        // use binary search for finding all k < valid_i
        #pragma omp parallel for schedule(dynamic)
        for(int k=1; k<=j; ++k) {
            long long sum = 0;
            auto it = upper_bound(valid_i.begin(), valid_i.end(), k);

            #pragma omp parallel for reduction(+:sum)
            for (; it != valid_i.end(); ++it) {
                int i = *it;
                sum += dp[k][i-2] * dp[i][j-1];
            }
            dp[k][j] += sum;
        }

        if (verbose) {
            std::cout << "j="<<j<<": ";
            for(int i=1;i<=j;i++) std::cout<<dp[i][j]<<" ";
            std::cout<<"\n";
        }
    }

    return dp[1][n];
}

int main() {
    
    // omp_set_nested(1);
    omp_set_num_threads(8);

    std::string test0 = "ACAGU";
    // Test cases: 16S rRNA    
    std::string test1 = "AUUCUGGUUGAUCCUGCCAGAGGCCGCUGCUAUCCGGCUGGGACUAAGCCAUGCGAGUCAAGGGGCUUGUAUCCCUUCGGGGAUGCAAGCACCGGCGGACGGCUCAGUAACACGUGGACAACCUGCCCUCGGGUGGGGGAUAACCCCGGGAAACUGGGGCUAAUCCCCCAUAGGGGAUGGGUACUGGAAUGUCCCAUCUCCGAAAGCGCUUAGCGCCCGAGGAUGGGUCUGCGGCGGAUUAGGUUGUUGGUGGGGUAACGGCCCACCAAGCCGAAGAUCCGUACGGGCCAUGAGAGUGGGAGCCCGGAGAUGGACCCUGAGACACGGGUCCAGGCCCUACGGGGCGCAGCAGGCGCGAAACCUCCGCAAUGCGGGAAACCGCGACGGGGUCAGCCGGAGUGCUCGCGCAUCGCGCGGGCUGUCGGGGUGCCUAAAAAGCACCCCACAGCAAGGGCCGGGCAAGGCCGGUGGCAGCCGCCGCGGUAAUACCGGCGGCCCGAGUGGCGGCCACUUUUAUUGGGCCUAAAGCGUCCGUAGCCGGGCUGGUAAGUCCUCCGGGAAAUCUGGCGGCUUAACCGUCAGACUGCCGGAGGAUACUGCCAGCCUAGGGACCGGGAGAGGCCGGGGGUAUUCCCGGAGUAGGGGUGAAAUCCUGUAAUCCCGGGAGGACCACCUGUGGCGAAGGCGCCCGGCUGGAACGGGUCCGACGGUGAGGGACGAAGGCCAGGGGAGCGAACCGGAUUAGAUACCCGGGUAGUCCUGGCUGUAAACGAUGCGGACUAGGUGUCACCGAAGCUACGAGCUUCGGUGGUGCCGGAGGGAAGCCGUUAAGUCCGCCGCCUGGGGAGUACGGCCGCAAGGCUGAAACUUAAAGGAAUUGGCGGGGGAGCACUACAACGGGUGGAGCCUGCGGUUUAAUUGGAUUCAACGCCGGGAAGCUUACCGGGGGAGACAGCGGGAUGAAGGUCGGGCUGAAGACCUUACCAGACUAGCUGAGAGGUGGUGCAUGGCCGCCGUCAGUUCGUACUGUGAAGCAUCCUGUUAAGUCAGGCAACGAGCGAGACCCGCGCCCCCAGUUGCCAGCGGUUCCCUUCGGGGAAGCCGGGCACACUGGGGGGACUGCCGGCGCUAAGCCGGAGGAAGGUGCGGGCAACGGCAGGUCCGUAUGCCCCGAAUCCCCCGGGCUACACGCGGGCUACAAUGGCCGGGACAAUGGGUACCGACCCCGAAAGGGGUAGGUAAUCCCCUAAACCCGGUCUAACCUGGGAUCGAGGGCUGCAACUCGCCCUCGUGAACCUGGAAUCCGUAGUAAUCGCGCCUCAAAAUGGCGCGGUGAAUACGUCCCUGCUCCUUGCACACACCGCCCGUCAAGCCACCCGAGUGGGCCAGGGGCGAGGGGGUGGCCCUAGGCCACCUUCGAGCCCAGGGUCCGCGAGGGGGGCUAAGUCGUAACAAGGUAGCCGUAGGGGAAUCUGCGGCUGGAUCACCUCCU";
    std::string test2 = "UUCCCUGAAGAGUUUGAUCCUGGCUCAGCGCGAACGCUGGCGGCGUGCCUAACACAUGCAAGUCGUGCGCAGGCUCGCUCCCUCUGGGAGCGGGUGCUGAGCGGCAAACGGGUGAGUAACACGUGGGUAACCUACCCCCAGGAGGGGGAUAACCCCGGGAAACCGGGGCUAAUACCCCAUAAAGCCGCCCGCCACUAAGGCGAGGCGGCCAAAGGGGGCCUCUGGGCUCUGCCCAAGCUCCCGCCUGGGGAUGGGCCCGCGGCCCAUCAGGUAGUUGGUGGGGUAACGGCCCACCAAGCCUAUGACGGGUAGCCGGCCUGAGAGGGUGGCCGGCCACAGCGGGACUGAGACACGGCCCGCACCCCUACGGGGGGCAGCAGUGGGGAAUCGUGGGCAAUGGGCGAAAGCCUGACCCCGCGACGCCGCGUGGGGGAAGAAGCCCUGCGGGGUGUAAACCCCUGUCGGGGGGGACGAAGGGACUGUGGGUUAAUAGCCCACAGUCUUGACGGUACCCCCAGAGGAAGGGACGGCUAACUACGUGCCAGCAGCCGCGGUAAUACGUAGGUCCCGAGCGUUGCGCGAAGUCACUGGGCGUAAAGCGUCCGCAGCCGGUCGGGUAAGCGGGAUGUCAAAGCCCACGGCUCAACCGUGGAAUGGCAUCCCGAACUGCCCGACUUGAGGCACGCCCGGGCAGGCGGAAUUCCCGGGGUAGCGGUGAAAUGCGUAGAUCUCGGGAGGAACACCGAAGGGGAAGCCAGCCUGCUGGGGCUGUCCUGACGGUCAGGGACGAAAGCCGGGGGAGCGAACCGGAUUAGAUACCCGGGUAGUCCCGGCCGUAAACCAUGGGCGCUAGGGCUUGUCCCUUUGGGGCAGGCUCGCAGCUAACGCGUUAAGCGCCCCGCCUGGGGAGUACGGGCGCAAGCCUGAAACUCAAAGGAAUUGGCGGGGGCCCGCACAACCGGUGGAGCGUCUGGUUCAAUUCGAUGCUAACCGAAGAACCUUACCCGGGCUUGACAUGCCGGGGAGACUCCGCGAAAGCGGAGUUGUGGAAGUCUCUGACUUCCCCCCGGCACAGGUGGUGCAUGGCCGUCGUCAGCUCGUGUCGUGAGAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCCUGCCCCUAGUUGCUACCCCGAGAGGGGAGCACUCUAGGGGGACCGCCGGCGAUAAGCCGGAGGAAGGGGGGGAUGACGUCAGGUCAGUAUGCCCUUUAUGCCCGGGGCCACACAGGCGCUACAGUGGCCGGGACAAUGGGAAGCGACCCCGCAAGGGGGAGCUAAUCCCAGAAACCCGGUCAUGGUGCGGAUUGGGGGCUGAAACUCGCCCCCAUGAAGCCGGAAUCGGUAGUAACGGGGUAUCAGCGAUGUCCCCGUGAAUACGUUCUCGGGCCUUGCACACACCGCCCGUCACGCCACGGAAGUCGGUCCGGCCGGAAGUCCCCGAGCUAACCGGCCCUUUUUGGGCCGGGGGCAGGGGCCGAUGGCCGGGCCGGCGACUGGGGCGAAGUCGUAACAAGGUAGCCGUAGGGGAACCUGC";
    // Test Cases: GRP II 
    std::string test3 = "AGUUUAGUGGUAAAAGUGUGAUUCGUUCUAUUAUCCCUUAAAUAGUUAAAGGGUCCUUCGGUUUGAUUCGUAUUCCGAUCAAAAACUUGAUUUCUAAAAAGGAUUUAAUCCUUUUCCUCUCAAUGACAGAUUCGAGAACAAAUACACAUUCUCGUGAUUUGUAUCCAAGGGUCACUUAGACAUUGAAAAAUUGGAUUAUGAAAUUGCGAAACAUAAUUUUGGAAUUGGAUCAAUACUUCCAAUUGAAUAAGUAUGAAUAAAGGAUCCAUGGAUGAAGAUAGAAAGUUGAUUUCUAAUCGUAACUAAAUCUUCAAUUUCUUAUUUGUAAAGAAGAAAUUGAAGCAAAAUAGCUAUUAAACGAUGACUUUGGUUUACUAGAGACAUCAACAUAUUGUUUUAGCUCGGUGGAAACAAAACCCUUUUCCUCAGGAUCCUAUUAAAUAGAAAUAGAGAACGAAAUAACUAGAAAGGUUGUUAGAAUCCCCUCUUCUAGAAGGAUCAUCUACAAAGCUAUUCGUUUUAUCUGUAUUCAGACCAAAAGCUGACAUAGAUGUUAUGGGUAGAAUUCUUUUUUUUUUUCGAAUUUUGUUCACAUCUUAGAUCUAUAAAUUGACUCAUCUCCAUAAAGGAGCCGAAUGAAACCAAAGUUUCAUGUUCGGUUUUGAAUUAGAGACGUUAAAAAUAAUGAAUCGACGUCGACUAUAACCC";
    std::string test4 = "GCUAGGGAUAACAGGGUGCGACCUGCCAAGCUGCACAAUUCAAUGUGGUUAGAAAACCAACUUGGAAUCCAAUCUCCAUGAGCCUACCAUCACAUGCGUUCUAGGGUUAACCUGAAGGUGUGAAGCUGAUGGGAAAAAGUAACCCAAACUGUAUGUGACAGUGAGGGGGCAGGCUAGAUUCCUAUGGGCAAUGUAAAUGAACACUCAUCUGAGGCAUCGUGACCCUAUCACAUCUAGUUAAUUGUGAGAGAAUCUUAUGUCUCUGUUUCAUAAGAUUGAUUGGACAAUUUCUCACCGGUGUAAAGAGUGGUCCUAAGGGAAUCAUCGAAAGUGAAUUGUGCGGAACAGGGCAAGCCCCAUAGGCUCCUUCGGGAGUGAGCGAAGCAAUUCUCUCUAUCGCCUAGUGGGUAAAAGACAGGGCAAAAAGCGAAUGCCAGUUAGUAAUAGACUGGAUAGGGUGAAUAACCUAACCUGAAAGGGUGCAGACUUGCUCAUGGGCGUGGGAAAUCAGAUUUCGUGAAUACACCAGCAUUCAAGAGUUUCCAUGCUUGAGCCGUGUGCGGUGAAAGUCGCAUGCACGGUUCUACUGGGGGGAAAGCCUGAGAGGGCCUACCUAUCCAACUUU";
    // std::string test5 = "AGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUAAGGCAUCAAACCCUGCAUGGGAGCGGAUGCCGUGUAGUCCAAAGACUUCUUUGGCACUA";

    std::string test5 = test2 + test2; // 3k
    std::string test6 = test5 + test2; // 4.5k

    //std::cout << "total(\"" << test0 << "\") = " << total0(test0, true) << "\n";
    //std::cout << "total(\"" << test0 << "\") = " << total1(test0, true) << "\n";
    //std::cout << "total(\"" << test2 << "\") = " << total0(test2, false) << "\n";

    //std::cout << "total(\"" << test1 << "\") = " << total(test1, false) << "\n";
    //std::cout << "total(\"" << test4 << "\") = " << total(test4, false) << "\n";

    /// CORRECTNESS CHECK
    // std::cout << "total(\"" << test3 << "\") = " << total1(test3, false) << "\n";
    // std::cout << "total(\"" << test3 << "\") = " << total14(test3, false) << "\n";

    std::cout << "total(\"" << test3 << "\") = " << total1(test3, false) << "\n";
    std::cout << "total(\"" << test3 << "\") = " << total14(test3, false) << "\n";

    // std::cout << "total(\"" << test6 << "\") = " << total0(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total1(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total2(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total3(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total4(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total5(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total6(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total7(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total14(test6, false) << "\n";

    std::cout << "total(\"" << test6 << "\") = " << total14(test6, false) << "\n";

    return 0;
}
