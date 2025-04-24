// g++ --std=c++11 rna_mp.cpp -o rna_mp -fopenmp
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
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

        // for (const auto& map : thread_maps) {
        //     for (const auto& kv : map) {
        //         int k = kv.first;
        //         long long val = kv.second;
        //         dp[k][j] += val;
        //     }            
        // }

        // #pragma omp parallel for
        // for (int k = 1; k <= j; ++k) {
        //     for (int t = 0; t < num_threads; ++t) {
        //         if (thread_maps[t].count(k))
        //             dp[k][j] += thread_maps[t][k];
        //     }
        // }

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
    std::cout << "total(\"" << test3 << "\") = " << total1(test3, false) << "\n";
    std::cout << "total(\"" << test3 << "\") = " << total8(test3, false) << "\n";

    // std::cout << "total(\"" << test6 << "\") = " << total0(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total1(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total2(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total3(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total4(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total5(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total6(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total7(test6, false) << "\n";
    std::cout << "total(\"" << test6 << "\") = " << total8(test6, false) << "\n";

    return 0;
}
