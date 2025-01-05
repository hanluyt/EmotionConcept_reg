require('R.matlab')
require(tidyverse)
require(dplyr)
require(ggplot2)
require(rstatix)
library(philentropy)
library(cluster)
library(jsonlite)

########################################################################
################# 1000 spin tests for occipital coding ################
########################################################################
num = 1000
permute <- read_csv('data/occ_permute1000_allVAE.csv')
t_permute <- rep(0, num)

angry_top10_percept_f256 = readMat('data/ANN_mapping/angry_top10_percept_f256.mat')$angry.top10.percept.f256 %>%
  as.data.frame

neutral_top10_percept_f256 = readMat('data/ANN_mapping/neutral_top10_percept_f256.mat')$neutral.top10.percept.f256 %>%
  as.data.frame

ANN <- colMeans(angry_top10_percept_f256) - colMeans(neutral_top10_percept_f256)


BrainScore <- function(vec1, vec2, num_bins=40){
  # normalization
  vec1_norm = 2 * (vec1 - min(vec1)) / (max(vec1) - min(vec1)) - 1
  vec2_norm = 2 * (vec2 - min(vec2)) / (max(vec2) - min(vec2)) - 1
  
  #prob distribution
  bins = cut(vec1_norm, breaks = num_bins)
  counts = table(bins)
  probs_vec1 = counts / sum(counts)
  
  # ANN prob distribution
  bins = cut(vec2_norm, breaks = num_bins)
  counts = table(bins)
  probs_vec2 = counts / sum(counts)
  
  # JS divergence
  probs_combine <- rbind(probs_vec1, probs_vec2)
  JS <- JSD(probs_combine, unit = "log2") 
  return(1-JS)
}

for (i in 1:num){
  brain <- unlist(permute[i,])
  score <- BrainScore(brain, ANN)
  t_permute[i] <- score
}


angry_neutral_occipital = readMat('data/ANN_mapping/angry_mu_seed100_occipital.mat')$angry.mu.seed100.occipital %>%
  as.data.frame # [1332, 256]
occ <- colMeans(angry_neutral_occipital)

t0 <- BrainScore(occ, ANN)

p_permute <- sum(abs(t0) < abs(t_permute)) /num


## noise ceiling ##
noiseceiling <- function(data) {
  n <- nrow(data)
  correlations <- numeric(n)  
  
  for (i in 1:n) {
    # remove each participants
    other_indices <- setdiff(1:n, i)
    # compute the mean pattern of the other participants
    mean_pattern <- colMeans(data[other_indices, ])
    # the correlation of each participant with the mean pattern of the other participants
    correlations[i] <- BrainScore(unlist(data[i, ]), mean_pattern)
  }
  
  return(mean(correlations))
}
brain_noiseceiling <- noiseceiling(angry_neutral_occipital) # 


##############################################################################################
################# Brain score(brain, CRPN) – Brain score(brain, PerceptPath) ################
#############################################################################################
###### CRPN
angry_p = readMat('data/ANN_mapping/angry_top10_percept_f256.mat')$angry.top10.percept.f256 %>% 
  as.data.frame
neutral_p = readMat('data/ANN_mapping/neutral_top10_percept_f256.mat')$neutral.top10.percept.f256 %>% 
  as.data.frame
angry_neutral_p <- colMeans(angry_p) - colMeans(neutral_p)

######## PerceptPath
angry_p_pure = readMat('data/ANN_mapping/angry_top10_percept_pure_f256.mat')$angry.top10.percept.pure.f256 %>% 
  as.data.frame
neutral_p_pure = readMat('data/ANN_mapping/neutral_top10_percept_pure_f256.mat')$neutral.top10.percept.pure.f256 %>% 
  as.data.frame
angry_neutral_p_pure <- colMeans(angry_p_pure) - colMeans(neutral_p_pure)

######## Brain data
angry_neutral_occipital = readMat('data/ANN_mapping/angry_mu_seed100_occipital.mat')$angry.mu.seed100.occipital %>%
  as.data.frame # [1332, 256]


######## 1000 bootstraps to compute 95%CI of the difference between Brain score for CRPN and PerceptPath #######
bootstrap_js_diff <- function(angry_neutral_occipital, 
                              angry_neutral_percept, angry_neutral_percept_pure, 
                              num_bins=40,n_iter = 1000) {
  diffs = numeric(n_iter)
  ### normalization for ANN ###
  angry_neutral_percept = 2 * (angry_neutral_percept - min(angry_neutral_percept)) / (max(angry_neutral_percept) - min(angry_neutral_percept)) - 1
  angry_neutral_percept_pure = 2 * (angry_neutral_percept_pure - min(angry_neutral_percept_pure)) / (max(angry_neutral_percept_pure) - min(angry_neutral_percept_pure)) - 1
  
  for (i in 1:n_iter) {
    # Sample with replacement
    idx = sample(nrow(angry_neutral_occipital), replace = TRUE)
    sample_occi = angry_neutral_occipital[idx, ]
    # Average 
    mean_occi = colMeans(sample_occi)
    # [-1, 1]
    mean_occi = 2 * (mean_occi - min(mean_occi)) / (max(mean_occi) - min(mean_occi)) - 1
    ########## JS divergence #########
    # perceptual
    bins = cut(angry_neutral_percept, breaks = num_bins)
    counts = table(bins)
    probs_percept = counts / sum(counts)
    
    # perceptual pure
    bins = cut(angry_neutral_percept_pure, breaks = num_bins)
    counts = table(bins)
    probs_percept_pure = counts / sum(counts)
    
    # occipital
    bins = cut(mean_occi, breaks = num_bins)
    counts = table(bins)
    probs_occi = counts / sum(counts)
    
    p_occi <- rbind(probs_percept, probs_occi)
    JS_p_occi <- JSD(p_occi, unit = "log2") 
    score_p_occi <- 1 - JS_p_occi
    
    pure_occi <- rbind(probs_percept_pure, probs_occi)
    JS_pure_occi <- JSD(pure_occi, unit = "log2") 
    score_pure_occi <- 1 - JS_pure_occi
    
    delta_p <- score_p_occi - score_pure_occi
    
    diffs[i] = delta_p
    
  }
  # 95%CI
  ci_p = quantile(diffs, c(0.025, 0.975))
  return(list(diffs = diffs, ci_p = ci_p))
}

result <- bootstrap_js_diff(angry_neutral_occipital, 
                            angry_neutral_p, angry_neutral_p_pure)


occ <- colMeans(angry_neutral_occipital)
S_crpn <- BrainScore(angry_neutral_p, occ)
S_percept <- BrainScore(angry_neutral_p_pure, occ)
S_diff <- S_crpn - S_percept

