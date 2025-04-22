// g++ --std=c++11 rna_mp.cpp -o rna_mp -fopenmp
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
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

// Trying to remove the n^2 mem space overhead -- BAD IDEA
// long long total4(std::string s, bool verbose) {
//     int n = s.size();

//     s = " " + s; // now 1-based

//     // Allowed pairs as two-character strings.
//     std::unordered_set<std::string> allowed = {"AU", "UA", "CG", "GC", "GU", "UG"};

//     // Create a 3D DP table with dimensions (n+1) x (n+1), initialized to 0.
//     std::vector<std::vector<std::vector<long long>>> dp
//     (n + 1, std::vector<std::vector<long long>>
//         (
//             n + 1, std::vector<long long>(n + 1, 0)
//         )
//     );

//     std::vector<std::vector<long long>> dp2(n + 1, std::vector<long long>(n + 1, 0));


//     // follow LP pseudocode:
//     // Base cases: dp[j][j-1] = 1 for each j=1..n (empty subsequence)
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int k = 1; k <= n; ++k) {
//         for (int j = 1; j <= n; ++j) {
//             dp[k][j][j-1] = 1;
//         }
//     }

//     // left to right
//     for (int j = 1; j <= n; j++) {
//         // Parallelize the loop over i for the current span.


//         // ALLOCATING 2D mem space costs O(n^2) time, for every i
//         // std::vector<std::vector<long long>> dp_tmp(n + 1, std::vector<long long>(n + 1, 0));

//         #pragma omp parallel for schedule(static)
//         for (int k = 1; k <= n; ++k) {
//             for (int i = 1; i <= j; ++i) {            
//                 // Case 1: skip (j is unpaired)
//                 dp[k][i][j] = dp[k][i][j-1];
//             }
//         }

//         #pragma omp parallel for schedule(static)

//         for (int i = 1; i <= j; ++i) {            

//             // Case 2: {i-1} pair with j? 
//             std::string pair = {s[i-1], s[j]};
//             if (allowed.count(pair)) {    

//                 for (int k = 1; k <= i-1; ++k) {  // k<=i-1 not k<=i-2 !

//                     dp[k][i][j] = dp2[k][i-2] * dp2[i][j-1]; //single-element write lock
//                 }
//             }
//         }

//         #pragma omp parallel for schedule(static)        
//         // parallel sum
//         for (int k = 1; k <= j; ++k) {            
//             for (int i=k+1; i<=j;i++)
//                 dp2[k][j] += dp[k][i][j];
//         }

//         if (verbose) {
//             std::cout << "j=" << j << std::endl;
//             for (int i = 1; i <= j; ++i) 
//                 std::cout << i<<"-"<<j<<":"<< dp2[i][j] << " ";
//             std::cout << std::endl;
//         }
//     }
//     return dp2[1][n];
// }

int main() {
    
    // omp_set_nested(1);

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
    // std::cout << "total(\"" << test3 << "\") = " << total2(test3, false) << "\n";
    // std::cout << "total(\"" << test3 << "\") = " << total3(test3, false) << "\n";

    // std::cout << "total(\"" << test6 << "\") = " << total0(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total1(test6, false) << "\n";
    // std::cout << "total(\"" << test6 << "\") = " << total2(test6, false) << "\n";
    std::cout << "total(\"" << test6 << "\") = " << total3(test6, false) << "\n";

    return 0;
}
