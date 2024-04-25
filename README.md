# SAFE-FUZZ
## Description
Testing Automated Driving Systems (ADS) has become increasingly complex as autonomous driving technology develops. To overcome these challenges, various search-based and fuzz-based methods have been proposed. However, these methods often overlook the impact of vehicle configuration on autonomous vehicles. We have introduced a fuzz-based framework for searching unsafe vehicle configuration settings(VCSs) to address this issue. This framework effectively identifies the combination of vehicle configuration with the minor disturbance that can lead to the maximum unsafe configuration. We evaluated two weather scenarios using the CARLA simulator and proposed that employing the fuzz-based testing method effectively generates many unsafe configurations. We also analyzed these unsafe configurations and identified the most common combinations of unsafe configurations.

This repository contains:

1. DataSet : all the raw data for the analyses (including two settings);
2. nocrash_runner.py is the file we use in the experiment to generate VCSs, which should replace the nocrash_runner.py from original [World On Rails](https://github.com/dotchen/WorldOnRails/blob/release/docs/INSTALL.md) repository.
   
## Contributions
1. We proposed that fuzzy testing in the field of ADS should consider changes in the vehicle itself, not just search environment elements of driving scenarios. 
2. We proposed SAFE-FUZZER, a fuzzing testing framework for finding unsafe VCSs under ADS's control.
3. We conducted experiments in two weather scenarios, and the results showed that our method can effectively generate unsafe VCSs and VCS ranges.

## Prerequisite
* CARLA 0.9.10
* Python : You should refer to the World On Rail to set up environment and the WordOnRail2.0 README to run program;

## Contributers